"""Tests for the thorn_cpp workflow orchestration.

Loads the orchestration module via the synthetic-package mechanism used
at runtime, then exercises _run_with_validation, delegate_to_child, and
coordinate with mocked validation and agent prompts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._discovery import _ensure_package, _load_thorn_module
from thorn.core._messages import UserMessage
from thorn.core._provider import FinishChunk, MockProvider, TextChunk
from thorn.core.errors import SkillError

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

_bt = _load_thorn_module(_THORN_DIR, _THORN_DIR / "build_tools.py")
_parse_doctest_output = _bt._parse_doctest_output
_format_test_summary = _bt._format_test_summary
_discover_test_executables = _bt._discover_test_executables
ExecutableResult = _bt.ExecutableResult


def _text_response(text: str):
    return [TextChunk(text=text), FinishChunk(reason="stop")]


def _module_developer_cls() -> type[Agent]:
    """Retrieve the ModuleDeveloper class from the Agent registry."""
    cls = Agent._registry.get("ModuleDeveloper")
    assert cls is not None, "ModuleDeveloper not found in Agent registry"
    return cls


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


class TestModuleDeveloper:
    """Tests for the ModuleDeveloper class properties."""

    def test_str_returns_developer_at_module(self):
        cls = _module_developer_cls()
        agent = cls(module="parser.lexer")
        assert str(agent) == "developer@parser.lexer"

    def test_str_root_module(self):
        cls = _module_developer_cls()
        agent = cls(module="main")
        assert str(agent) == "developer@main"

    def test_validation_rules_include_build_and_test(self):
        cls = _module_developer_cls()
        rules = cls._collect_validation_rules()
        assert "build" in rules
        assert "test" in rules

    def test_instance_file_access_non_root(self):
        """A non-root developer gets write access to its header, source,
        and test files."""
        cls = _module_developer_cls()
        agent = cls(module="parser")
        rules = agent._instance_file_access()

        write_patterns = [r.pattern for r in rules]
        assert any("parser.h" in p for p in write_patterns)
        assert any("parser.cpp" in p for p in write_patterns)
        assert any("tests" in p for p in write_patterns)

    def test_instance_file_access_root(self):
        """The root (main) developer gets write access to main.cpp and
        test files, but not a header (main has no header)."""
        cls = _module_developer_cls()
        agent = cls(module="main")
        rules = agent._instance_file_access()

        write_patterns = [r.pattern for r in rules]
        assert any("main.cpp" in p for p in write_patterns)
        assert any("tests" in p for p in write_patterns)
        assert not any("main.h" in p for p in write_patterns)

    def test_system_prompts_include_resolved_paths_for_non_root(self):
        """A non-root developer's system prompts should include its
        concrete header and source file paths."""
        cls = _module_developer_cls()
        agent = cls(module="eval")
        rendered = agent._render_system_prompts()
        joined = "\n".join(rendered)

        assert "src/eval.h" in joined, (
            "Expected resolved header path src/eval.h in system prompts"
        )
        assert "src/eval.cpp" in joined, (
            "Expected resolved source path src/eval.cpp in system prompts"
        )

    def test_system_prompts_include_resolved_paths_for_nested(self):
        """A nested module developer should get paths reflecting the
        hierarchy (not a repeated directory, e.g. src/parser/lexer.h,
        NOT src/parser/lexer/lexer.h)."""
        cls = _module_developer_cls()
        agent = cls(module="parser.lexer")
        rendered = agent._render_system_prompts()
        joined = "\n".join(rendered)

        assert "src/parser/lexer.h" in joined
        assert "src/parser/lexer.cpp" in joined

    def test_system_prompts_include_resolved_paths_for_root(self):
        """The root (main) developer should see its source path and an
        indication that it has no header."""
        cls = _module_developer_cls()
        agent = cls(module="main")
        rendered = agent._render_system_prompts()
        joined = "\n".join(rendered)

        assert "main.cpp" in joined
        assert "no header" in joined.lower()

    def test_filesystem_convention_shows_single_segment_example(self):
        """The shared filesystem convention prompt should include an
        example for single-segment (top-level) modules to prevent agents
        from incorrectly mapping e.g. 'eval' to 'src/eval/eval.h'."""
        cls = _module_developer_cls()
        agent = cls(module="eval")
        rendered = agent._render_system_prompts()
        joined = "\n".join(rendered)

        assert "src/foo.h" in joined, (
            "Convention should show a top-level module example like src/foo.h"
        )


class TestCoordinate:
    """Tests for the top-level coordinate() entry point."""

    @pytest.fixture(autouse=True)
    def _ctx(self):
        self.provider = MockProvider()
        ctx = ExecutionContext(provider=self.provider)
        token = set_context(ctx)
        yield
        reset_context(token)

    async def test_creates_module_developer(self):
        """coordinate() should create a ModuleDeveloper, not a Coordinator."""
        created_agents: list[Agent] = []

        async def capture(agent, task, **kwargs):
            created_agents.append(agent)
            return "done"

        with patch.object(_orch, "_run_with_validation", side_effect=capture):
            await _orch.coordinate("build the thing")

        assert len(created_agents) == 1
        assert type(created_agents[0]).__name__ == "ModuleDeveloper"
        assert created_agents[0].module == "main"

    async def test_custom_module(self):
        """coordinate() respects the module parameter."""
        created_agents: list[Agent] = []

        async def capture(agent, task, **kwargs):
            created_agents.append(agent)
            return "done"

        with patch.object(_orch, "_run_with_validation", side_effect=capture):
            await _orch.coordinate("fix bug", module="parser")

        assert created_agents[0].module == "parser"

    async def test_skill_error_wrapped_as_runtime_error(self):
        """coordinate() wraps SkillError from the developer into
        RuntimeError with a descriptive message."""
        async def raise_skill(agent, task, **kwargs):
            raise SkillError("cannot fix this")

        with patch.object(_orch, "_run_with_validation", side_effect=raise_skill):
            with pytest.raises(RuntimeError, match="developer@main"):
                await _orch.coordinate("do it")


class TestDelegateToChild:
    """Tests for the delegate_to_child() delegation tool."""

    @pytest.fixture(autouse=True)
    def _ctx(self):
        self.provider = MockProvider()
        parent = _module_developer_cls()(module="main")
        ctx = ExecutionContext(provider=self.provider, agent=parent)
        token = set_context(ctx)
        yield
        reset_context(token)

    async def test_creates_module_developer_for_child(self):
        """delegate_to_child() should create a ModuleDeveloper for the
        child module."""
        created_agents: list[Agent] = []

        async def capture(agent, task, **kwargs):
            created_agents.append(agent)
            return "done"

        with (
            patch.object(
                _orch, "list_submodules", return_value=["parser", "repl"],
            ),
            patch.object(_orch, "_run_with_validation", side_effect=capture),
        ):
            await _orch.delegate_to_child("parser", "implement the parser")

        assert len(created_agents) == 1
        assert type(created_agents[0]).__name__ == "ModuleDeveloper"
        assert created_agents[0].module == "parser"

    async def test_invalid_child_raises_value_error(self):
        """delegate_to_child() raises ValueError for unknown children."""
        with patch.object(
            _orch, "list_submodules", return_value=["parser"],
        ):
            with pytest.raises(ValueError, match="not a submodule"):
                await _orch.delegate_to_child("unknown", "task")

    async def test_skill_error_wrapped_as_runtime_error(self):
        """delegate_to_child() wraps SkillError into RuntimeError."""
        async def raise_skill(agent, task, **kwargs):
            raise SkillError("child failed")

        with (
            patch.object(
                _orch, "list_submodules", return_value=["parser"],
            ),
            patch.object(_orch, "_run_with_validation", side_effect=raise_skill),
        ):
            with pytest.raises(RuntimeError, match="developer@parser"):
                await _orch.delegate_to_child("parser", "task")


# ---------------------------------------------------------------------------
# Doctest output parsing (build_tools)
# ---------------------------------------------------------------------------

# Realistic doctest outputs used across multiple tests.

_DOCTEST_ALL_PASS = """\
[doctest] doctest version is "2.4.11"
[doctest] run with "--help" for options
===============================================================================
[doctest] test cases:  5 |  5 passed | 0 failed | 0 skipped
[doctest] assertions: 12 | 12 passed | 0 failed | 0 skipped
"""

_DOCTEST_ONE_FAILURE_GCC = """\
[doctest] doctest version is "2.4.11"
[doctest] run with "--help" for options
===============================================================================
tests/repl_test.cpp:67:
TEST CASE:  REPL evaluates simple expressions

tests/repl_test.cpp:76: ERROR: CHECK( io.get_output() == "5\\n" ) is NOT correct!
  values: CHECK( "5.000000\\n" == "5\\n" )

===============================================================================
[doctest] test cases: 16 | 15 passed | 1 failed | 0 skipped
[doctest] assertions: 19 | 18 passed | 1 failed | 0 skipped
"""

_DOCTEST_ONE_FAILURE_MSVC = """\
[doctest] doctest version is "2.4.11"
[doctest] run with "--help" for options
===============================================================================
tests\\repl_test.cpp(67):
TEST CASE:  REPL evaluates simple expressions

tests\\repl_test.cpp(76): ERROR: CHECK( io.get_output() == "5\\n" ) is NOT correct!
  values: CHECK( "5.000000\\n" == "5\\n" )

===============================================================================
[doctest] test cases: 16 | 15 passed | 1 failed | 0 skipped
[doctest] assertions: 19 | 18 passed | 1 failed | 0 skipped
"""

_DOCTEST_TWO_FAILURES = """\
[doctest] doctest version is "2.4.11"
[doctest] run with "--help" for options
===============================================================================
tests/repl_test.cpp:67:
TEST CASE:  REPL evaluates simple expressions

tests/repl_test.cpp:76: ERROR: CHECK( io.get_output() == "5\\n" ) is NOT correct!
  values: CHECK( "5.000000\\n" == "5\\n" )

===============================================================================
tests/repl_test.cpp:90:
TEST CASE:  REPL handles assignment

tests/repl_test.cpp:98: ERROR: CHECK( io.get_output() == "x = 42\\n" ) is NOT correct!
  values: CHECK( "x = 42.000000\\n" == "x = 42\\n" )

===============================================================================
[doctest] test cases: 16 | 14 passed | 2 failed | 0 skipped
[doctest] assertions: 19 | 17 passed | 2 failed | 0 skipped
"""


class TestParseDoctestOutput:
    """Tests for _parse_doctest_output."""

    def test_all_passing(self):
        result = _parse_doctest_output("expr_test", 0, _DOCTEST_ALL_PASS)

        assert result.passed
        assert result.test_count == 5
        assert result.pass_count == 5
        assert result.fail_count == 0
        assert result.assertion_count == 12
        assert result.failures == []

    def test_single_failure_gcc_format(self):
        result = _parse_doctest_output("repl_test", 1, _DOCTEST_ONE_FAILURE_GCC)

        assert not result.passed
        assert result.test_count == 16
        assert result.pass_count == 15
        assert result.fail_count == 1
        assert result.assertion_count == 19
        assert result.assertion_fail_count == 1

        assert len(result.failures) == 1
        case = result.failures[0]
        assert case.name == "REPL evaluates simple expressions"
        assert "repl_test.cpp" in case.location
        assert "67" in case.location

        assert len(case.assertions) == 1
        assertion = case.assertions[0]
        assert "CHECK" in assertion.expression
        assert "is NOT correct" in assertion.expression
        assert "5.000000" in assertion.values

    def test_single_failure_msvc_format(self):
        result = _parse_doctest_output("repl_test", 1, _DOCTEST_ONE_FAILURE_MSVC)

        assert not result.passed
        assert len(result.failures) == 1
        case = result.failures[0]
        assert case.name == "REPL evaluates simple expressions"
        assert "repl_test.cpp" in case.location
        assert "67" in case.location
        assert len(case.assertions) == 1

    def test_multiple_failures(self):
        result = _parse_doctest_output("repl_test", 1, _DOCTEST_TWO_FAILURES)

        assert not result.passed
        assert result.fail_count == 2
        assert len(result.failures) == 2

        names = {f.name for f in result.failures}
        assert "REPL evaluates simple expressions" in names
        assert "REPL handles assignment" in names

    def test_unparseable_output_nonzero_rc(self):
        """When output doesn't match doctest format, we still record a
        failure if the return code is non-zero."""
        result = _parse_doctest_output("mystery_test", 1, "segfault\n")

        assert not result.passed
        assert result.fail_count == 1
        assert result.failures == []
        assert result.raw_output == "segfault\n"

    def test_empty_output_rc_zero(self):
        result = _parse_doctest_output("empty_test", 0, "")

        assert result.passed
        assert result.test_count == 0
        assert result.failures == []


class TestFormatTestSummary:
    """Tests for _format_test_summary."""

    def test_all_passing(self):
        results = [
            ExecutableResult(
                name="expr_test", test_count=5, pass_count=5,
                assertion_count=12, assertion_pass_count=12,
            ),
            ExecutableResult(
                name="parser_test", test_count=3, pass_count=3,
                assertion_count=8, assertion_pass_count=8,
            ),
        ]
        summary = _format_test_summary(results)

        assert summary.startswith("[run_tests OK]")
        assert "2 executable(s)" in summary
        assert "8 test cases" in summary
        assert "all passed" in summary
        assert "FAILED" not in summary

    def test_one_failure(self):
        failure = _bt.TestCaseFailure(
            name="REPL evaluates simple expressions",
            location="tests/repl_test.cpp:67",
            assertions=[
                _bt.AssertionFailure(
                    location="tests/repl_test.cpp:76",
                    expression='CHECK( io.get_output() == "5\\n" ) is NOT correct!',
                    values='CHECK( "5.000000\\n" == "5\\n" )',
                ),
            ],
        )
        results = [
            ExecutableResult(
                name="expr_test", test_count=5, pass_count=5,
                assertion_count=12, assertion_pass_count=12,
            ),
            ExecutableResult(
                name="repl_test", test_count=16, pass_count=15, fail_count=1,
                assertion_count=19, assertion_pass_count=18,
                assertion_fail_count=1, failures=[failure],
            ),
        ]
        summary = _format_test_summary(results)

        assert summary.startswith("[run_tests FAILED]")
        assert "REPL evaluates simple expressions" in summary
        assert "tests/repl_test.cpp:67" in summary
        assert "5.000000" in summary
        assert "Passed: expr_test" in summary

    def test_raw_output_fallback(self):
        """When failure parsing yields no structured failures, the raw
        output is included as a fallback."""
        results = [
            ExecutableResult(
                name="crash_test", fail_count=1,
                raw_output="Segmentation fault (core dumped)",
            ),
        ]
        summary = _format_test_summary(results)

        assert "FAILED: crash_test" in summary
        assert "raw output follows" in summary
        assert "Segmentation fault" in summary


class TestDiscoverTestExecutables:
    """Tests for _discover_test_executables."""

    def test_empty_directory(self, tmp_path: Path):
        assert _discover_test_executables(tmp_path) == []

    def test_nonexistent_directory(self, tmp_path: Path):
        assert _discover_test_executables(tmp_path / "nope") == []

    def test_finds_exe_files(self, tmp_path: Path):
        (tmp_path / "expr_test.exe").write_bytes(b"")
        (tmp_path / "parser_test.exe").write_bytes(b"")
        (tmp_path / "calc.exe").write_bytes(b"")

        found = _discover_test_executables(tmp_path)
        stems = {p.stem for p in found}

        assert "expr_test" in stems
        assert "parser_test" in stems
        assert "calc" not in stems

    def test_finds_in_subdirectories(self, tmp_path: Path):
        debug = tmp_path / "Debug"
        debug.mkdir()
        (debug / "repl_test.exe").write_bytes(b"")

        found = _discover_test_executables(tmp_path)
        assert len(found) == 1
        assert found[0].stem == "repl_test"

    def test_deduplicates_by_stem(self, tmp_path: Path):
        """If both .exe and extensionless versions exist, only one is returned."""
        (tmp_path / "expr_test.exe").write_bytes(b"")
        ext_less = tmp_path / "expr_test"
        ext_less.write_bytes(b"")

        found = _discover_test_executables(tmp_path)
        assert len(found) == 1
