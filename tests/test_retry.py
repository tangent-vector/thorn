"""Tests for thorn._retry — bounded retry utilities."""

from __future__ import annotations

import pytest

from thorn.core._retry import bound_retries
from thorn.core.errors import LoopLimitError


class TestBoundRetries:
    def test_break_on_first_attempt(self):
        collected = []
        for attempt in bound_retries(5):
            collected.append(attempt)
            break
        assert collected == [0]

    def test_break_on_later_attempt(self):
        collected = []
        for attempt in bound_retries(5):
            collected.append(attempt)
            if attempt == 2:
                break
        assert collected == [0, 1, 2]

    def test_yields_all_indices_before_raising(self):
        collected = []
        with pytest.raises(LoopLimitError) as exc_info:
            for attempt in bound_retries(4):
                collected.append(attempt)
        assert collected == [0, 1, 2, 3]
        assert exc_info.value.rounds == 4

    def test_single_attempt_success(self):
        for attempt in bound_retries(1):
            assert attempt == 0
            break

    def test_single_attempt_exhaustion(self):
        with pytest.raises(LoopLimitError) as exc_info:
            for attempt in bound_retries(1):
                pass
        assert exc_info.value.rounds == 1

    def test_error_message_includes_count(self):
        with pytest.raises(LoopLimitError, match="7 attempts"):
            for _ in bound_retries(7):
                pass

    def test_default_max_attempts_is_three(self):
        collected = []
        with pytest.raises(LoopLimitError) as exc_info:
            for attempt in bound_retries():
                collected.append(attempt)
        assert collected == [0, 1, 2]
        assert exc_info.value.rounds == 3

    def test_body_exception_propagates_not_loop_limit(self):
        with pytest.raises(ValueError, match="boom"):
            for _ in bound_retries(5):
                raise ValueError("boom")
