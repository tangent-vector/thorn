"""Built-in tools for thorn agents.

Each tool is an ordinary Python function.  They are exposed to agents
via ``wrap_function()`` and can also be called directly from user code.

File-access tools (``read_file``, ``write_file``, ``list_directory``)
enforce the active ``FileAccessPolicy`` from the current
``ExecutionContext``, when one is set.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


def _enforce_access(path: str, required_name: str) -> None:
    """Check the active file-access policy, if any.

    Imports are deferred so that the module can still be loaded in
    contexts where no ``ExecutionContext`` is active (e.g. direct
    scripting use of ``read_file``).
    """
    from thorn._context import get_context
    from thorn._file_access import FileAccessLevel, check_access

    try:
        ctx = get_context()
    except RuntimeError:
        return

    policy = ctx.file_access_policy
    workspace = ctx.workspace_root
    if policy is None or workspace is None:
        return

    required = FileAccessLevel[required_name]
    check_access(path, required, policy=policy, workspace=workspace)


async def read_file(path: str) -> str:
    """Read and return the contents of a file at the given path."""
    _enforce_access(path, "READ")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    _enforce_access(path, "WRITE")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


async def list_directory(path: str = ".") -> list[str]:
    """List entries in a directory.  Returns a list of names."""
    _enforce_access(path, "READ")
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    entries = sorted(entry.name for entry in p.iterdir())

    from thorn._context import get_context
    from thorn._file_access import FileAccessLevel, resolve_for_check

    try:
        ctx = get_context()
    except RuntimeError:
        return entries

    policy = ctx.file_access_policy
    workspace = ctx.workspace_root
    if policy is None or workspace is None:
        return entries

    return policy.filter_listing(entries, resolve_for_check(path, workspace))


async def run_shell(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr.

    Not included in ``ALL_BUILTIN_TOOLS`` — add it explicitly to
    specific agents or ``prompt()`` calls when needed.
    """
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
    answer = await loop.run_in_executor(None, lambda: input(f"\n? {question}\n> "))
    return answer


ALL_BUILTIN_TOOLS = [read_file, write_file, list_directory, ask_user]
