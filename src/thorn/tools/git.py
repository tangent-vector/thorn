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
from pathlib import Path
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


def _git_identity_env() -> dict[str, str] | None:
    """Build a subprocess environment with git author/committer identity.

    Reads ``git_user_name`` and ``git_user_email`` from the current
    agent's metadata.  Returns ``None`` when no identity is available,
    in which case callers should let the subprocess inherit the ambient
    environment (which may or may not have git identity configured).
    """
    from thorn.core._context import get_context

    try:
        ctx = get_context()
    except RuntimeError:
        return None

    agent = ctx.agent
    if agent is None:
        return None

    name = agent.metadata.get("git_user_name")
    email = agent.metadata.get("git_user_email")
    if not name and not email:
        return None

    env = os.environ.copy()
    if name:
        env["GIT_AUTHOR_NAME"] = name
        env["GIT_COMMITTER_NAME"] = name
    if email:
        env["GIT_AUTHOR_EMAIL"] = email
        env["GIT_COMMITTER_EMAIL"] = email
    return env


async def _run_git(
    *args: str,
    cwd: str | None = None,
    check: bool = True,
) -> tuple[int, str]:
    """Run a git command asynchronously, returning (returncode, combined output).

    When the current agent has ``git_user_name`` / ``git_user_email``
    in its metadata, those are injected as ``GIT_AUTHOR_*`` and
    ``GIT_COMMITTER_*`` environment variables so commits succeed even
    when the system git config has no identity set.

    Raises ``GitError`` when *check* is True and the process exits non-zero.
    """
    cmd = ["git", *args]
    log.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=_git_identity_env(),
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


def _inject_url_credentials(url: str) -> str:
    """Rewrite a git HTTPS URL to embed credentials from the current agent.

    Uses ``metadata.project`` (project service name) to find the forge
    service on the ambient :class:`~thorn.runtime.Runtime`, then embeds
    the current HTTPS password or token.  GitHub uses the
    ``x-access-token`` username; GitLab uses ``oauth2``.
    """
    from thorn.core._context import get_context
    from thorn.tools.forge import (
        ForgeHostService,
        GitHubForgeService,
        ProjectService,
    )

    try:
        ctx = get_context()
    except RuntimeError:
        return url

    agent = ctx.agent
    if agent is None or ctx.runtime is None:
        return url

    project_name = agent.metadata.get("project")
    if not project_name:
        return url

    try:
        project_svc = ctx.runtime.get_service(project_name)
    except KeyError:
        return url

    if not isinstance(project_svc, ProjectService):
        return url

    try:
        forge_svc = ctx.runtime.get_service(project_svc.forge_name)
    except KeyError:
        return url

    if not isinstance(forge_svc, ForgeHostService):
        return url

    token = forge_svc.git_https_password()

    if url.startswith("https://"):
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port_suffix = f":{parsed.port}" if parsed.port else ""
        user = "x-access-token" if isinstance(forge_svc, GitHubForgeService) else "oauth2"
        rewritten = parsed._replace(
            netloc=f"{user}:{token}@{host}{port_suffix}",
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


def _safe_paths_under_repo(repo_resolved: str, paths: list[str]) -> list[str]:
    """Return *paths* unchanged if each resolves under *repo_resolved*.

    Rejects absolute paths and paths that escape the repository via ``..``.
    """
    root = Path(repo_resolved).resolve()
    for raw in paths:
        if os.path.isabs(raw):
            raise ValueError(
                "git_add paths must be relative to the repository, "
                f"got absolute path: {raw!r}",
            )
        joined = (root / raw).resolve()
        try:
            joined.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"git_add path escapes repository root: {raw!r}",
            ) from exc
    return paths


async def _remaining_changes_note(resolved_repo: str) -> str:
    """Non-empty string to append after a successful commit if the tree is not clean."""
    _, output = await _run_git("status", "--short", cwd=resolved_repo)
    text = output.strip()
    if not text:
        return ""
    return (
        "\n\nRemaining changes after commit (unstaged and/or untracked):\n"
        + text
    )


# ---------------------------------------------------------------------------
# @tool functions
# ---------------------------------------------------------------------------


@tool
async def git_clone(
    remote_url: str,
    local_path: str,
    bare: bool = False,
) -> str:
    """Clone a git repository, or fetch updates if it already exists.

    By default, performs a normal clone that produces a working tree
    (checked-out files you can edit directly).  Pass ``bare=True`` for
    a bare clone suitable for managing multiple worktrees.

    Credentials are injected transparently from the agent's metadata
    when available.

    Returns a confirmation message with the local path.
    """
    resolved = _resolve_tool_path(local_path)
    authenticated_url = _inject_url_credentials(remote_url)

    if os.path.isdir(resolved):
        _, output = await _run_git("fetch", "--all", cwd=resolved)
        return f"Fetched updates in {local_path}\n{output}".strip()

    clone_args = ["clone"]
    if bare:
        clone_args.append("--bare")
    clone_args.extend([authenticated_url, resolved])

    _, output = await _run_git(*clone_args)
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
async def git_add(repo_path: str, paths: list[str] | None = None) -> str:
    """Stage changes in the repository index.

    When *paths* is omitted (or ``None``), runs ``git add -A`` in the
    repository — stage all new, modified, and deleted paths. Prefer
    passing an explicit *paths* list when you know which files changed,
    to avoid accidentally staging unrelated files.

    Paths are relative to the repository root. Path segments must not
    escape the repository (no ``..`` traversal to outside the repo).

    Args:
        repo_path: Root of the git repository or worktree.
        paths: Files or directories to stage; omit for ``git add -A``.
    """
    resolved = _resolve_tool_path(repo_path)
    if paths is None:
        _, output = await _run_git("add", "-A", cwd=resolved)
    else:
        if not paths:
            return (
                "Error: when provided, paths must be non-empty, "
                "or omit paths entirely to stage all changes."
            )
        _safe_paths_under_repo(resolved, paths)
        _, output = await _run_git("add", "--", *paths, cwd=resolved)
    return output.strip() if output.strip() else "Staged changes."


@tool
async def git_commit(repo_path: str, message: str) -> str:
    """Create a commit from the current index.

    Runs ``git commit -m <message>`` only. Stage changes first with
    ``git_add``. If nothing is staged, git exits with an error
    (same as the CLI).

    On success, if the working tree still has unstaged or untracked
    changes, that status is appended so you can notice missed files.

    Args:
        repo_path: Root of the git repository or worktree.
        message: Commit message.
    """
    resolved = _resolve_tool_path(repo_path)
    _, output = await _run_git("commit", "-m", message, cwd=resolved)
    body = output.strip()
    remainder = await _remaining_changes_note(resolved)
    return body + remainder


@tool
async def git_push(
    repo_path: str,
    branch_name: str,
    remote: str = "origin",
) -> str:
    """Push a branch to a remote repository.

    Pushes the specified *branch_name* to *remote* (default ``origin``).
    Authentication is handled transparently via the forge service
    registered for the agent's project (see ``_inject_url_credentials``).
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
async def git_fetch(
    repo_path: str,
    remote: str = "origin",
) -> str:
    """Fetch updates from a remote without modifying the working tree.

    Retrieves new commits, branches, and tags from *remote* (default
    ``origin``).  Works in both bare repositories and worktrees.
    """
    resolved = _resolve_tool_path(repo_path)
    _, output = await _run_git("fetch", remote, cwd=resolved)
    return f"Fetched from {remote}\n{output}".strip()


@tool
async def git_pull(
    repo_path: str,
    remote: str = "origin",
    branch: str | None = None,
) -> str:
    """Pull changes from a remote into the current branch.

    Runs ``git pull <remote> [<branch>]``.  When *branch* is not
    specified, pulls the tracking branch.  Typically used inside a
    worktree to incorporate upstream changes (e.g. reviewer-pushed
    fixup commits or base-branch updates).
    """
    resolved = _resolve_tool_path(repo_path)
    args = ["pull", remote]
    if branch is not None:
        args.append(branch)
    _, output = await _run_git(*args, cwd=resolved)
    return f"Pulled from {remote}\n{output}".strip()


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

    The worktree directory must **not** be located inside the bare
    repository path; nesting it there produces a broken checkout.
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
    git_add,
    git_commit,
    git_push,
    git_fetch,
    git_pull,
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
    "git_add",
    "git_commit",
    "git_push",
    "git_fetch",
    "git_pull",
    "git_status",
    "git_diff",
    "git_worktree_add",
    "git_worktree_remove",
    "git_log",
    "GIT_TOOLS",
]
