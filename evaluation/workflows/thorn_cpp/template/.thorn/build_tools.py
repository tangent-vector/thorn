"""Build tools for the calc example project.

These are auto-discovered by thorn from the .thorn/ directory.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from thorn import tool

PROJECT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_DIR / "build"


async def _run(cmd: str, cwd: Path | None = None) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd or PROJECT_DIR,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace") if stdout else ""
    return proc.returncode or 0, output


# ---------------------------------------------------------------------------
# Doctest output parsing
# ---------------------------------------------------------------------------


@dataclass
class AssertionFailure:
    """A single failed assertion within a doctest TEST_CASE."""

    location: str
    expression: str
    values: str


@dataclass
class TestCaseFailure:
    """A failed TEST_CASE containing one or more assertion failures."""

    name: str
    location: str
    assertions: list[AssertionFailure] = field(default_factory=list)


@dataclass
class ExecutableResult:
    """Parsed result of running a single test executable."""

    name: str
    test_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    assertion_count: int = 0
    assertion_pass_count: int = 0
    assertion_fail_count: int = 0
    failures: list[TestCaseFailure] = field(default_factory=list)
    raw_output: str = ""

    @property
    def passed(self) -> bool:
        return self.fail_count == 0


_TC_SUMMARY_RE = re.compile(
    r"\[doctest\] test cases:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed"
)
_ASSERT_SUMMARY_RE = re.compile(
    r"\[doctest\] assertions:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed"
)
_TEST_CASE_RE = re.compile(r"TEST CASE:\s+(.*\S)")
_ERROR_RE = re.compile(r"(?:ERROR|FATAL ERROR):\s+(.+)")
_VALUES_RE = re.compile(r"^\s+values:\s+(.*)")


def _parse_doctest_output(name: str, rc: int, output: str) -> ExecutableResult:
    """Parse doctest test runner output into a structured result."""
    result = ExecutableResult(name=name, raw_output=output)

    tc_match = _TC_SUMMARY_RE.search(output)
    if tc_match:
        result.test_count = int(tc_match.group(1))
        result.pass_count = int(tc_match.group(2))
        result.fail_count = int(tc_match.group(3))
    elif rc != 0:
        result.fail_count = 1

    assert_match = _ASSERT_SUMMARY_RE.search(output)
    if assert_match:
        result.assertion_count = int(assert_match.group(1))
        result.assertion_pass_count = int(assert_match.group(2))
        result.assertion_fail_count = int(assert_match.group(3))

    if result.passed:
        return result

    blocks = re.split(r"^={10,}\s*$", output, flags=re.MULTILINE)
    for block in blocks:
        tc_match = _TEST_CASE_RE.search(block)
        if not tc_match:
            continue

        location = ""
        before_tc = block[: tc_match.start()]
        for line in reversed(before_tc.strip().splitlines()):
            stripped = line.strip()
            if stripped:
                location = stripped.rstrip(":")
                break

        case_failure = TestCaseFailure(
            name=tc_match.group(1).strip(),
            location=location,
        )

        lines = block.splitlines()
        for i, line in enumerate(lines):
            error_match = _ERROR_RE.search(line)
            if not error_match:
                continue

            for marker in ("FATAL ERROR:", "ERROR:"):
                pos = line.find(marker)
                if pos != -1:
                    err_location = line[:pos].strip().rstrip(":")
                    break
            else:
                err_location = ""

            expression = error_match.group(1).strip()

            values = ""
            if i + 1 < len(lines):
                val_match = _VALUES_RE.match(lines[i + 1])
                if val_match:
                    values = val_match.group(1).strip()

            case_failure.assertions.append(
                AssertionFailure(
                    location=err_location,
                    expression=expression,
                    values=values,
                )
            )

        if case_failure.assertions:
            result.failures.append(case_failure)

    return result


def _discover_test_executables(
    build_dir: Path = BUILD_DIR,
) -> list[Path]:
    """Find test executables (``*_test`` / ``*_test.exe``) under *build_dir*."""
    if not build_dir.exists():
        return []

    seen_stems: set[str] = set()
    executables: list[Path] = []

    for candidate in sorted(build_dir.rglob("*_test.exe")):
        if candidate.is_file() and candidate.stem not in seen_stems:
            seen_stems.add(candidate.stem)
            executables.append(candidate)

    for candidate in sorted(build_dir.rglob("*_test")):
        if (
            candidate.is_file()
            and candidate.suffix == ""
            and candidate.stem not in seen_stems
            and os.access(candidate, os.X_OK)
        ):
            seen_stems.add(candidate.stem)
            executables.append(candidate)

    return executables


def _format_test_summary(results: list[ExecutableResult]) -> str:
    """Format test results into a concise, actionable summary."""
    total_tests = sum(r.test_count for r in results)
    total_passed = sum(r.pass_count for r in results)
    total_failed = sum(r.fail_count for r in results)
    total_assertions = sum(r.assertion_count for r in results)
    total_assert_passed = sum(r.assertion_pass_count for r in results)
    total_assert_failed = sum(r.assertion_fail_count for r in results)

    all_passed = all(r.passed for r in results)

    if all_passed:
        parts = [f"[run_tests OK] {len(results)} executable(s)"]
        if total_tests:
            parts.append(
                f", {total_tests} test cases"
                f", {total_assertions} assertions"
            )
        parts.append(" -- all passed.")
        return "".join(parts)

    lines: list[str] = ["[run_tests FAILED]"]

    summary_parts = [f"Summary: {len(results)} executable(s)"]
    if total_tests:
        summary_parts.append(
            f", {total_tests} test cases"
            f" ({total_passed} passed, {total_failed} failed)"
        )
    if total_assertions:
        summary_parts.append(
            f", {total_assertions} assertions"
            f" ({total_assert_passed} passed, {total_assert_failed} failed)"
        )
    lines.append("".join(summary_parts))

    passed_names = [r.name for r in results if r.passed]
    if passed_names:
        lines.append(f"\nPassed: {', '.join(passed_names)}")

    for result in results:
        if result.passed:
            continue

        lines.append(f"\nFAILED: {result.name}")

        if result.failures:
            for case in result.failures:
                loc = f" ({case.location})" if case.location else ""
                lines.append(f"  TEST CASE: {case.name}{loc}")
                for assertion in case.assertions:
                    lines.append(f"    {assertion.expression}")
                    if assertion.values:
                        lines.append(f"    values: {assertion.values}")
        elif result.raw_output:
            lines.append(
                "  (could not parse failure details; raw output follows)"
            )
            for raw_line in result.raw_output.strip().splitlines():
                lines.append(f"  {raw_line}")

    return "\n".join(lines)


@tool
async def configure() -> str:
    """Run CMake configure step, creating the build directory if needed."""
    BUILD_DIR.mkdir(exist_ok=True)
    rc, output = await _run(f'cmake -S "{PROJECT_DIR}" -B "{BUILD_DIR}"')
    if rc != 0:
        return f"[configure FAILED, exit {rc}]\n{output}"
    return f"[configure OK]\n{output}"


@tool
async def build() -> str:
    """Build the calc project (always re-configures to pick up new files)."""
    cfg = await configure()
    if "[configure FAILED" in cfg:
        return cfg

    rc, output = await _run(f'cmake --build "{BUILD_DIR}"')
    if rc != 0:
        return f"[build FAILED, exit {rc}]\n{output}"
    return f"[build OK]\n{output}"


@tool
async def build_tests() -> str:
    """Build all test executables (re-configures first to pick up new files)."""
    cfg = await configure()
    if "[configure FAILED" in cfg:
        return cfg

    rc, output = await _run(f'cmake --build "{BUILD_DIR}"')
    if rc != 0:
        return f"[build_tests FAILED, exit {rc}]\n{output}"
    return f"[build_tests OK]\n{output}"


@tool
async def run_tests() -> str:
    """Build and run all tests, returning a structured pass/fail summary.

    Runs each test executable directly and parses doctest output to produce
    a concise report.  On failure the report includes the failing test case
    name, source location, assertion expression, and actual-vs-expected
    values so you can identify the problem without additional investigation.
    """
    build_result = await build_tests()
    if "[build_tests FAILED" in build_result or "[configure FAILED" in build_result:
        return build_result

    executables = _discover_test_executables()
    if not executables:
        return "[run_tests OK] No test executables found."

    results: list[ExecutableResult] = []
    for exe in executables:
        rc, output = await _run(f'"{exe}"')
        results.append(_parse_doctest_output(exe.stem, rc, output))

    return _format_test_summary(results)


@tool
async def clean() -> str:
    """Remove the build directory entirely."""
    import shutil
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        return f"Removed {BUILD_DIR}"
    return "Nothing to clean."


@tool
async def run_calc(input_text: str = "") -> str:
    """Run the calc binary with optional input piped to stdin."""
    candidates = list(BUILD_DIR.rglob("calc.exe")) + list(BUILD_DIR.rglob("calc"))
    exe = None
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            exe = c
            break

    if exe is None:
        return "[error] calc binary not found — have you run build()?"

    proc = await asyncio.create_subprocess_exec(
        str(exe),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate(input=input_text.encode())
    output = stdout.decode(errors="replace") if stdout else ""
    return output
