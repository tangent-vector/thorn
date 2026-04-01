"""Build tools for the calc example project.

These are auto-discovered by thorn from the .thorn/ directory.
"""

from __future__ import annotations

import asyncio
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
    """Build all test executables then run them via ctest.

    Returns a pass/fail summary with failure details.
    """
    build_result = await build_tests()
    if "[build_tests FAILED" in build_result or "[configure FAILED" in build_result:
        return build_result

    rc, output = await _run(
        f'ctest --test-dir "{BUILD_DIR}" -C Debug --output-on-failure'
    )
    if "No tests were found" in output:
        return "[run_tests OK] No tests found."
    if rc != 0:
        return f"[run_tests FAILED, exit {rc}]\n{output}"
    return f"[run_tests OK]\n{output}"


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
    import os

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
