"""Git operations as ``@tool``-decorated functions for thorn agents.

Provides subprocess-based Git tools that can be exposed to agents for
repository management: clone/fetch, branch creation, commit, push,
status, diff, and worktree management.

All tools are async, using ``asyncio.create_subprocess_exec`` to avoid
blocking the event loop during potentially long-running git operations.

Usage::

    from thorn.tools import git
    agent = Agent(..., tools=[git.GIT_TOOLS, ...])

Adapted from ``thorn-bot/src/thorn_bot/_git.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal
from urllib.parse import urlparse, urlunparse

from thorn.core._func import tool

log = logging.getLogger(__name__)


class GitError(Exception):
    """A git subprocess exited with a non-zero return code."""

    def __init__(self, args: list[str], returncode: int, output: str) -> None:
        self.args_list = args
        self.returncode = returncode
        self.output = output
        cmd_str = " ".join(args)
        super().__init__(f"git command failed (exit {returncode}): {cmd_str}\n{output}")


async def _run_git(
    *args: str,
    cwd: str | None = None,
    check: bool = True,
) -> tuple[int, str]:
    """Run a git command asynchronously, returning (returncode, combined output).

    Raises ``GitError`` when *check* is True and the process exits non-zero.
    """
    cmd = ["git", *args]
    log.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    raw_stdout, _ = await proc.communicate()
    output = raw_stdout.decode(errors="replace") if raw_stdout else ""
    returncode = proc.returncode or 0

    if check and returncode != 0:
        raise GitError(cmd, returncode, output)

    return returncode, output


# ---------------------------------------------------------------------------
# Credential injection
# ---------------------------------------------------------------------------


def _resolve_env_reference(value: str) -> str:
    """Resolve a value that may reference an environment variable.

    If *value* starts with ``$``, the remainder is treated as an
    environment variable name whose value is returned.  Otherwise
    *value* is returned as-is.

    Raises ``ValueError`` if the referenced variable is not set.
    """
    if value.startswith("$"):
        env_name = value[1:]
        result = os.environ.get(env_name)
        if result is None:
            raise ValueError(
                f"Environment variable {env_name!r} is not set "
                f"(referenced by agent metadata)"
            )
        return result
    return value


def _inject_url_credentials(url: str) -> str:
    """Rewrite a git HTTPS URL to embed credentials from the current agent.

    Looks up ``access_token`` in the agent's ``metadata`` (via the
    ambient ``ExecutionContext``).  If the token value starts with
    ``$``, it is resolved as an environment variable reference.

    The URL is rewritten to ``https://oauth2:{token}@host/path`` so
    that git operations authenticate transparently.  Non-HTTPS URLs
    and agents without an ``access_token`` are returned unchanged.
    """
    from thorn.core._context import get_context

    try:
        ctx = get_context()
    except RuntimeError:
        return url

    agent = ctx.agent
    if agent is None:
        return url

    token_ref = agent.metadata.get("access_token")
    if not token_ref:
        return url

    token = _resolve_env_reference(token_ref)

    if url.startswith("https://"):
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port_suffix = f":{parsed.port}" if parsed.port else ""
        rewritten = parsed._replace(
            netloc=f"oauth2:{token}@{host}{port_suffix}",
        )
        return urlunparse(rewritten)

    return url


# ---------------------------------------------------------------------------
# Workspace-aware path resolution
# ---------------------------------------------------------------------------


def _resolve_tool_path(path: str) -> str:
    """Resolve a tool path argument against the active workspace.

    Returns an absolute path string.  Deferred import avoids circular
    imports at module-load time.
    """
    from thorn.core._context import resolve_path
    return str(resolve_path(path))


# ---------------------------------------------------------------------------
# @tool functions
# ---------------------------------------------------------------------------


@tool
async def git_clone(remote_url: str, local_path: str) -> str:
    """Clone a git repository (bare) or fetch updates if it already exists.

    Uses ``--bare`` clones so that worktrees can be created from the
    clone for parallel work on multiple branches.  Credentials are
    injected transparently from the agent's metadata when available.

    Returns a confirmation message with the local path.
    """
    resolved = _resolve_tool_path(local_path)
    authenticated_url = _inject_url_credentials(remote_url)

    if os.path.isdir(resolved):
        _, output = await _run_git("fetch", "--all", cwd=resolved)
        return f"Fetched updates in {local_path}\n{output}".strip()

    _, output = await _run_git("clone", "--bare", authenticated_url, resolved)
    return f"Cloned {remote_url} -> {local_path}\n{output}".strip()


@tool
async def git_branch(
    repo_path: str,
    branch_name: str,
    start_point: str = "HEAD",
) -> str:
    """Create and check out a new branch in a repository.

    Creates the branch from *start_point* (default ``HEAD``) and
    switches to it.  Fails if the branch already exists.
    """
    resolved = _resolve_tool_path(repo_path)
    _, output = await _run_git(
        "checkout", "-b", branch_name, start_point, cwd=resolved,
    )
    return f"Created and checked out branch '{branch_name}'\n{output}".strip()


@tool
async def git_commit(repo_path: str, message: str) -> str:
    """Stage all changes and create a commit.

    Equivalent to ``git add -A && git commit -m <message>``.
    Returns the commit summary.
    """
    resolved = _resolve_tool_path(repo_path)
    await _run_git("add", "-A", cwd=resolved)
    _, output = await _run_git("commit", "-m", message, cwd=resolved)
    return output.strip()


@tool
async def git_push(
    repo_path: str,
    branch_name: str,
    remote: str = "origin",
) -> str:
    """Push a branch to a remote repository.

    Pushes the specified *branch_name* to *remote* (default ``origin``).
    If the remote URL requires authentication and the agent has an
    ``access_token`` in its metadata, the push will authenticate
    transparently.
    """
    resolved = _resolve_tool_path(repo_path)
    _, output = await _run_git("push", remote, branch_name, cwd=resolved)
    return f"Pushed {branch_name} to {remote}\n{output}".strip()


@tool
async def git_status(repo_path: str) -> str:
    """Show the working tree status of a repository.

    Returns the output of ``git status --short``, which shows
    modified, added, deleted, and untracked files.
    """
    resolved = _resolve_tool_path(repo_path)
    _, output = await _run_git("status", "--short", cwd=resolved)
    if not output.strip():
        return "Working tree clean — no changes."
    return output.strip()


@tool
async def git_diff(
    repo_path: str,
    staged: bool = False,
) -> str:
    """Show the diff of changes in a repository.

    When *staged* is False (default), shows unstaged changes.
    When *staged* is True, shows changes that have been staged
    (``git diff --cached``).
    """
    resolved = _resolve_tool_path(repo_path)
    args = ["diff"]
    if staged:
        args.append("--cached")
    _, output = await _run_git(*args, cwd=resolved)
    if not output.strip():
        qualifier = "staged " if staged else ""
        return f"No {qualifier}changes."
    return output.strip()


@tool
async def git_worktree_add(
    bare_repo: str,
    worktree_path: str,
    branch_name: str,
    start_point: str = "HEAD",
) -> str:
    """Create a git worktree with a new branch from a bare repository.

    Useful for working on multiple branches in parallel.  The
    worktree is created at *worktree_path* with a new branch
    named *branch_name* starting from *start_point*.
    """
    resolved_repo = _resolve_tool_path(bare_repo)
    resolved_wt = _resolve_tool_path(worktree_path)
    _, output = await _run_git(
        "worktree", "add", "-b", branch_name,
        resolved_wt, start_point,
        cwd=resolved_repo,
    )
    return f"Created worktree at {worktree_path} on branch '{branch_name}'\n{output}".strip()


@tool
async def git_worktree_remove(bare_repo: str, worktree_path: str) -> str:
    """Remove a git worktree.

    Cleans up the worktree directory and its administrative files
    in the bare repository.
    """
    resolved_repo = _resolve_tool_path(bare_repo)
    resolved_wt = _resolve_tool_path(worktree_path)
    _, output = await _run_git(
        "worktree", "remove", resolved_wt, cwd=resolved_repo,
    )
    return f"Removed worktree {worktree_path}\n{output}".strip()


@tool
async def git_log(
    repo_path: str,
    max_count: int = 10,
    format: Literal["oneline", "short", "medium", "full"] = "oneline",
) -> str:
    """Show recent commit history.

    Returns the last *max_count* commits in the specified *format*.
    """
    resolved = _resolve_tool_path(repo_path)
    fmt_flag = f"--format={format}"
    _, output = await _run_git(
        "log", fmt_flag, f"--max-count={max_count}", cwd=resolved,
    )
    if not output.strip():
        return "No commits yet."
    return output.strip()


GIT_TOOLS: list[object] = [
    git_clone,
    git_branch,
    git_commit,
    git_push,
    git_status,
    git_diff,
    git_worktree_add,
    git_worktree_remove,
    git_log,
]
"""All git tools as a list, suitable for use in ``tools=[GIT_TOOLS, ...]``."""

__all__ = [
    "GitError",
    "git_clone",
    "git_branch",
    "git_commit",
    "git_push",
    "git_status",
    "git_diff",
    "git_worktree_add",
    "git_worktree_remove",
    "git_log",
    "GIT_TOOLS",
]
