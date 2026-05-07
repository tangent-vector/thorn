"""Exception hierarchy for thorn."""

from __future__ import annotations


class ThornError(Exception):
    """Base class for all thorn exceptions."""


class ProviderError(ThornError):
    """An LLM provider returned an unexpected error."""


class TransientProviderError(ProviderError):
    """A transport-level failure that is plausibly worth retrying.

    Covers connection resets, timeouts, mid-stream protocol errors,
    and retryable HTTP status codes (502/503/504).  The retry loop
    in :func:`thorn.core._loop._request_completion` distinguishes
    these from other :class:`ProviderError` instances so that it can
    keep re-trying past the ordinary ``max_failures`` cap, which is
    aimed at non-transport failures.

    Attributes:
        retry_after: When the failure was accompanied by a server
            hint about how long to wait (e.g. a ``Retry-After``
            header on a 503), the parsed value in seconds.  ``None``
            when no hint was provided.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitError(ProviderError):
    """The LLM provider rate-limited the request.

    Attributes:
        retry_after: The ``Retry-After`` header value in seconds, if
            the provider supplied one.  Callers should wait at
            least this long before re-issuing the request.  ``None``
            when no header was present.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderError):
    """Transient/rate-limit retries were exhausted without success.

    Raised by :func:`thorn.core._loop._request_completion` when the
    configured retry budget for :class:`TransientProviderError` and
    :class:`RateLimitError` has been used up.  Semantically, this
    says "we repeatedly tried to talk to the LLM provider and could
    not get a response" -- it is attributable to the provider /
    network, not to anything the agent or session did.

    Callers that distinguish *provider* failures from *agent*
    failures (notably the session scheduler, which skips its
    forward-progress strike increment on this exception) should
    catch :class:`ProviderUnavailableError` specifically rather
    than treating all :class:`ProviderError` instances alike.

    Attributes:
        attempts: The number of attempts that were made before
            giving up.  Useful for diagnostics/logging.
    """

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class LoopLimitError(ThornError):
    """The agent loop exceeded its maximum number of rounds."""

    def __init__(self, message: str, rounds: int) -> None:
        super().__init__(message)
        self.rounds = rounds


class LoopRepetitionError(LoopLimitError):
    """The agent loop repeated the same response pattern without progress.

    Raised before the ordinary maximum-round limit when consecutive
    provider turns produce the same text or the same failing tool-call
    round.  This is a loop-control failure rather than a provider
    transport failure.
    """

    def __init__(self, message: str, repetitions: int) -> None:
        super().__init__(message, repetitions)
        self.repetitions = repetitions


class SkillError(ThornError):
    """A prompt-based skill signalled failure via the ``raise_error`` tool.

    Attributes:
        detail: The error description provided by the agent.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class AgentFailureError(ThornError):
    """Too many consecutive non-transient provider failures in an agent loop.

    Raised when the retry budget for non-transient
    :class:`ProviderError` instances (e.g. repeated HTTP 4xx
    responses that are not rate limits) is exhausted.  This is
    distinct from :class:`ProviderUnavailableError`, which covers
    the transport-level / transient side.
    """

    def __init__(self, message: str, failures: int) -> None:
        super().__init__(message)
        self.failures = failures
