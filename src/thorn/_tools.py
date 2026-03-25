"""Built-in tools for thorn agents.

Each tool is an ordinary Python function.  They are exposed to agents
via ``wrap_function()`` and can also be called directly from user code.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


async def read_file(path: str) -> str:
    """Read and return the contents of a file at the given path."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


async def list_directory(path: str = ".") -> list[str]:
    """List entries in a directory.  Returns a list of names."""
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return sorted(entry.name for entry in p.iterdir())


async def run_shell(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace") if stdout else ""
    if proc.returncode != 0:
        return f"[exit code {proc.returncode}]\n{output}"
    return output


async def ask_user(question: str) -> str:
    """Ask the human user a question and return their response.

    In a non-interactive context this will raise an error.
    """
    loop = asyncio.get_running_loop()
    # run_in_executor so the blocking input() doesn't stall the event loop
    answer = await loop.run_in_executor(None, lambda: input(f"\n? {question}\n> "))
    return answer


ALL_BUILTIN_TOOLS = [read_file, write_file, list_directory, run_shell, ask_user]
