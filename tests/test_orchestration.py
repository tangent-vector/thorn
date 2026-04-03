"""Tests for _run_with_validation from the thorn_cpp workflow orchestration.

Loads the orchestration module via the synthetic-package mechanism used
at runtime, then exercises _run_with_validation with mocked validation
and agent prompts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from thorn._agent import Agent
from thorn._context import ExecutionContext, reset_context, set_context
from thorn._discovery import _ensure_package, _load_thorn_module
from thorn._messages import UserMessage
from thorn._provider import FinishChunk, MockProvider, TextChunk
from thorn.errors import SkillError

# ---------------------------------------------------------------------------
# Load the orchestration module via the same synthetic-package mechanism
# that thorn uses at runtime, so relative imports resolve correctly.
# ---------------------------------------------------------------------------

_THORN_DIR = (
    Path(__file__).resolve().parent.parent
    / "evaluation" / "workflows" / "thorn_cpp" / "template" / ".thorn"
)


def _load_orchestration():
    _ensure_package(_THORN_DIR)
    for sibling in sorted(_THORN_DIR.glob("*.py")):
        _load_thorn_module(_THORN_DIR, sibling)
    mod = _load_thorn_module(_THORN_DIR, _THORN_DIR / "orchestration.py")
    assert mod is not None, "Failed to load orchestration module"
    return mod


_orch = _load_orchestration()
_run_with_validation = _orch._run_with_validation


def _text_response(text: str):
    return [TextChunk(text=text), FinishChunk(reason="stop")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunWithValidation:
    """Tests for the validation retry loop."""

    @pytest.fixture(autouse=True)
    def _ctx(self):
        """Install a MockProvider-backed context for every test."""
        self.provider = MockProvider()
        ctx = ExecutionContext(provider=self.provider)
        token = set_context(ctx)
        yield
        reset_context(token)

    async def test_no_rules_returns_original_summary(self):
        """When no validation rules apply, the original summary is returned
        with no validation or re-prompting."""
        self.provider.canned_responses = [_text_response("task done")]

        agent = Agent()
        with patch.object(_orch, "effective_validation_rules", return_value=frozenset()):
            result = await _run_with_validation(agent, "do the thing")

        assert result == "task done"

    async def test_passes_first_try_returns_original_summary(self):
        """When validation passes on the first attempt, the original summary
        is returned without any additional prompts."""
        self.provider.canned_responses = [_text_response("implemented feature X")]

        agent = Agent()
        with (
            patch.object(
                _orch, "effective_validation_rules",
                return_value=frozenset({"build"}),
            ),
            patch.object(_orch, "_run_validation", return_value=[]),
        ):
            result = await _run_with_validation(agent, "implement feature X")

        assert result == "implemented feature X"

    async def test_retry_then_pass_returns_fresh_summary(self):
        """When validation fails on the first attempt and passes after a fix,
        the agent is asked for a final summary covering the whole task."""
        self.provider.canned_responses = [
            _text_response("implemented feature X"),
            _text_response("I fixed the build error"),
            _text_response("Added feature X with full test coverage"),
        ]

        validation_results = iter([
            [("build", "compilation error in foo.cpp")],
            [],
        ])

        agent = Agent()
        with (
            patch.object(
                _orch, "effective_validation_rules",
                return_value=frozenset({"build"}),
            ),
            patch.object(
                _orch, "_run_validation",
                side_effect=lambda _rules: validation_results.__next__(),
            ),
        ):
            result = await _run_with_validation(agent, "implement feature X")

        assert result == "Added feature X with full test coverage"

    async def test_retry_summary_prompt_text(self):
        """The final-summary prompt should ask for a summary of the work,
        not just the fixes."""
        user_prompts: list[str] = []

        class CapturingProvider(MockProvider):
            async def complete(self, system_prompts, tools, messages):
                for msg in messages:
                    if isinstance(msg, UserMessage):
                        user_prompts.append(msg.content)
                async for chunk in super().complete(
                    system_prompts, tools, messages,
                ):
                    yield chunk

        provider = CapturingProvider(canned_responses=[
            _text_response("original"),
            _text_response("fixed it"),
            _text_response("final summary"),
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)

        validation_results = iter([
            [("build", "error")],
            [],
        ])

        try:
            agent = Agent()
            with (
                patch.object(
                    _orch, "effective_validation_rules",
                    return_value=frozenset({"build"}),
                ),
                patch.object(
                    _orch, "_run_validation",
                    side_effect=lambda _rules: validation_results.__next__(),
                ),
            ):
                await _run_with_validation(agent, "implement feature X")
        finally:
            reset_context(token)

        last_prompts = [p for p in user_prompts if p not in ("",)]
        final_prompt = last_prompts[-1]
        assert "summary" in final_prompt.lower()
        assert "original task" in final_prompt.lower()

    async def test_exhausts_retries_raises_skill_error(self):
        """When validation keeps failing past max_retries, SkillError is raised."""
        self.provider.canned_responses = [
            _text_response("initial"),
            _text_response("fix attempt 1"),
            _text_response("fix attempt 2"),
        ]

        agent = Agent()
        with (
            patch.object(
                _orch, "effective_validation_rules",
                return_value=frozenset({"build"}),
            ),
            patch.object(
                _orch, "_run_validation",
                return_value=[("build", "still broken")],
            ),
        ):
            with pytest.raises(SkillError, match="still failing after 2 retries"):
                await _run_with_validation(agent, "do it", max_retries=2)

    async def test_multiple_retries_before_pass(self):
        """When validation fails multiple times before passing, the final
        summary prompt is still issued (not an intermediate fix response)."""
        self.provider.canned_responses = [
            _text_response("original work"),
            _text_response("fix 1"),
            _text_response("fix 2"),
            _text_response("comprehensive summary"),
        ]

        validation_results = iter([
            [("build", "error A")],
            [("test", "test failure")],
            [],
        ])

        agent = Agent()
        with (
            patch.object(
                _orch, "effective_validation_rules",
                return_value=frozenset({"build", "test"}),
            ),
            patch.object(
                _orch, "_run_validation",
                side_effect=lambda _rules: validation_results.__next__(),
            ),
        ):
            result = await _run_with_validation(agent, "complex task")

        assert result == "comprehensive summary"
