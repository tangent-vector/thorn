"""Exception hierarchy for thorn."""

from __future__ import annotations


class ThornError(Exception):
    """Base class for all thorn exceptions."""


class ProviderError(ThornError):
    """An LLM provider returned an unexpected error."""


class RateLimitError(ProviderError):
    """The LLM provider rate-limited the request."""


class LoopLimitError(ThornError):
    """The agent loop exceeded its maximum number of rounds."""

    def __init__(self, message: str, rounds: int) -> None:
        super().__init__(message)
        self.rounds = rounds


class SkillError(ThornError):
    """A prompt-based skill signalled failure via the ``raise_error`` tool.

    Attributes:
        detail: The error description provided by the agent.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class AgentFailureError(ThornError):
    """Too many consecutive provider failures during an agent loop."""

    def __init__(self, message: str, failures: int) -> None:
        super().__init__(message)
        self.failures = failures
