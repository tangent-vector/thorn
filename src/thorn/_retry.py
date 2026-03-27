"""Bounded retry utilities."""

from __future__ import annotations

from collections.abc import Iterator

from thorn.errors import LoopLimitError


def bound_retries(max_attempts: int = 3) -> Iterator[int]:
    """Yield attempt indices ``0 .. max_attempts-1``, then raise.

    Use with ``for``/``break`` — ``break`` out on success and the
    generator exits cleanly.  If the loop exhausts all attempts without
    a ``break``, :class:`LoopLimitError` is raised automatically::

        for attempt in bound_retries(3):
            do_work()
            if validate():
                break  # success — no exception

        # If we get here without breaking, LoopLimitError is raised.
    """
    for i in range(max_attempts):
        yield i
    raise LoopLimitError(
        f"Failed after {max_attempts} attempts",
        rounds=max_attempts,
    )
