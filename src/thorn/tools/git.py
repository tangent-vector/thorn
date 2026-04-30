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
import base64
import logging
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

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

    Resolution order:

    1. If the agent has a :class:`ForgeAccountConfig` for the project's
       forge (looked up via ``metadata["project"]``), use the account's
       ``git_user_name`` / ``git_user_email``.
    2. Otherwise fall back to ``agent.metadata["git_user_name"]`` /
       ``["git_user_email"]`` (legacy path).

    Returns ``None`` when no identity is available, in which case
    callers let the subprocess inherit the ambient environment.
    """
    from thorn.core._context import get_context

    try:
        ctx = get_context()
    except RuntimeError:
        return None

    agent = ctx.agent
    if agent is None:
        return None

    name, email = _resolve_git_identity(agent, ctx.runtime)
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


def _resolve_git_identity(
    agent: object,
    runtime: object,
) -> tuple[str, str]:
    """Return ``(name, email)`` for git commits, preferring account data.

    Tries the agent's :class:`ForgeAccountConfig` for the project's
    forge first, then falls back to ``agent.metadata`` keys.  Returns
    ``("", "")`` when nothing is available.
    """
    from thorn.core._account import resolve_account
    from thorn.tools.forge import ForgeHostService, ProjectService

    metadata: dict[str, object] = getattr(agent, "metadata", {})

    if runtime is not None:
        project_name = metadata.get("project")
        if isinstance(project_name, str) and project_name:
            try:
                project_svc = runtime.get_service(project_name)  # type: ignore[union-attr]
                if isinstance(project_svc, ProjectService):
                    forge_svc = runtime.get_service(project_svc.forge_name)  # type: ignore[union-attr]
                    if isinstance(forge_svc, ForgeHostService):
                        account = resolve_account(agent, forge_svc.name)  # type: ignore[arg-type]
                        acct_name = getattr(account, "git_user_name", "")
                        acct_email = getattr(account, "git_user_email", "")
                        if acct_name or acct_email:
                            return acct_name, acct_email
            except KeyError:
                pass

    name = str(metadata.get("git_user_name") or "")
    email = str(metadata.get("git_user_email") or "")
    return name, email


async def _run_git(
    *args: str,
    cwd: str | None = None,
    check: bool = True,
    auth: bool = False,
) -> tuple[int, str]:
    """Run a git command asynchronously, returning (returncode, combined output).

    When the current agent has ``git_user_name`` / ``git_user_email``
    in its metadata, those are injected as ``GIT_AUTHOR_*`` and
    ``GIT_COMMITTER_*`` environment variables so commits succeed even
    when the system git config has no identity set.

    When *auth* is True, additional ``GIT_CONFIG_*`` env vars are
    injected so that any HTTPS request git makes to the agent's
    project's forge carries an ``Authorization`` header derived from
    the agent's per-account credentials.  This avoids ever embedding
    tokens in URLs (which would otherwise leak into ``.git/config``)
    and is intended for git tools that perform network operations
    (clone, push, fetch, pull).

    Raises ``GitError`` when *check* is True and the process exits non-zero.
    """
    cmd = ["git", *args]
    log.debug("Running: %s (cwd=%s, auth=%s)", " ".join(cmd), cwd, auth)

    env = _git_identity_env()
    if auth:
        auth_env = _git_auth_env_for_current_agent()
        if auth_env:
            if env is None:
                env = os.environ.copy()
            _merge_git_config_env(env, auth_env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=env,
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


def _git_auth_env_for_current_agent() -> dict[str, str]:
    """Build subprocess env vars that authenticate git over HTTPS.

    Returns a mapping like::

        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://gitlab.example.com/.extraheader",
            "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic <base64(user:token)>",
        }

    When merged into a git subprocess's environment, modern git (>=
    2.31) reads these as ad-hoc ``-c key=value`` overrides.  The
    ``http.<URL>.extraheader`` form scopes the injected
    ``Authorization`` header to URLs that share the given prefix, so
    this never leaks into requests git makes to unrelated hosts (e.g.
    submodules from another forge).

    Resolves credentials from the agent's :class:`AccountConfig`
    for the project's forge (looked up via ``metadata["project"]``).
    Returns ``{}`` when there is no active agent context, no project
    metadata, no resolvable forge service, no matching account, or
    the forge service is of an unexpected type.  Callers should
    treat that as "let git fall through to whatever ambient
    credential helper the OS provides," which is the right behavior
    for both local development and unauthenticated public-repo
    clones.
    """
    from thorn.core._account import resolve_account
    from thorn.core._context import get_context
    from thorn.tools.forge import (
        ForgeHostService,
        GitHubForgeService,
        GitLabForgeService,
        ProjectService,
    )

    try:
        ctx = get_context()
    except RuntimeError:
        return {}

    agent = ctx.agent
    if agent is None or ctx.runtime is None:
        return {}

    project_name = agent.metadata.get("project")
    if not project_name:
        return {}

    try:
        project_svc = ctx.runtime.get_service(project_name)
    except KeyError:
        return {}

    if not isinstance(project_svc, ProjectService):
        return {}

    try:
        forge_svc = ctx.runtime.get_service(project_svc.forge_name)
    except KeyError:
        return {}

    if not isinstance(forge_svc, ForgeHostService):
        return {}

    try:
        account = resolve_account(agent, forge_svc.name)
    except KeyError:
        return {}

    try:
        token = forge_svc.git_https_password_for(account)
    except (KeyError, LookupError):
        return {}

    if not token:
        return {}

    if isinstance(forge_svc, GitHubForgeService):
        username = "x-access-token"
        url_prefix = _github_git_url_prefix(forge_svc.base_url)
    elif isinstance(forge_svc, GitLabForgeService):
        username = "oauth2"
        url_prefix = _https_origin_prefix(forge_svc.url)
    else:
        return {}

    if not url_prefix:
        return {}

    basic = base64.b64encode(f"{username}:{token}".encode()).decode()
    config_key = f"http.{url_prefix}.extraheader"
    config_value = f"AUTHORIZATION: basic {basic}"
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": config_key,
        "GIT_CONFIG_VALUE_0": config_value,
    }


def _github_git_url_prefix(api_base_url: str) -> str:
    """Derive the public ``https://<host>/`` prefix for a GitHub forge.

    GitHub.com's API lives at ``api.github.com`` while git URLs use
    ``github.com``; GitHub Enterprise generally uses the same host
    for both (with an ``/api/v3`` path suffix).  Mirrors the host
    derivation in :meth:`GitHubForgeService.clone_url_for`.
    """
    parsed = urlparse(api_base_url)
    host = parsed.hostname or "github.com"
    if host.startswith("api."):
        host = host[len("api."):]
    port_suffix = f":{parsed.port}" if parsed.port else ""
    return f"https://{host}{port_suffix}/"


def _https_origin_prefix(url: str) -> str:
    """Return ``https://<host>[:port]/`` for *url*, or ``""`` on parse failure."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return ""
    port_suffix = f":{parsed.port}" if parsed.port else ""
    return f"https://{host}{port_suffix}/"


def _merge_git_config_env(
    env: dict[str, str],
    auth_env: dict[str, str],
) -> None:
    """Merge auth_env's ``GIT_CONFIG_*`` entries into env.

    If env already carries a ``GIT_CONFIG_COUNT``-style block (e.g.
    inherited from the parent process), the new entries are appended
    after the existing ones rather than overwriting them, so a
    site-wide policy in ``GIT_CONFIG_*`` is preserved alongside our
    per-call auth header.
    """
    new_count = int(auth_env.get("GIT_CONFIG_COUNT", "0"))
    if new_count <= 0:
        return

    try:
        existing_count = int(env.get("GIT_CONFIG_COUNT", "0"))
    except ValueError:
        existing_count = 0

    for i in range(new_count):
        slot = existing_count + i
        env[f"GIT_CONFIG_KEY_{slot}"] = auth_env[f"GIT_CONFIG_KEY_{i}"]
        env[f"GIT_CONFIG_VALUE_{slot}"] = auth_env[f"GIT_CONFIG_VALUE_{i}"]
    env["GIT_CONFIG_COUNT"] = str(existing_count + new_count)


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
    """Clone a git repository into *local_path*.

    Always invokes ``git clone`` -- this tool deliberately does
    *not* fall back to ``git fetch`` when *local_path* already
    exists, since "fetch" and "clone" have very different semantics
    (fetch leaves the working tree untouched) and conflating them
    silently makes for surprising agent behavior.

    By default, performs a normal clone that produces a working tree
    (checked-out files you can edit directly).  Pass ``bare=True``
    for a bare clone suitable for managing multiple worktrees.

    If *local_path* already exists and is non-empty, ``git clone``
    will fail with ``destination path '...' already exists and is
    not an empty directory.``  In that case, treat the existing
    checkout as the source of truth: use ``git_fetch`` / ``git_pull``
    / ``git_branch`` etc. to bring it up to date rather than calling
    ``git_clone`` again.

    Credentials for HTTPS URLs are injected per-call via subprocess
    environment variables (a scoped ``http.<URL>.extraheader``
    config), so tokens never land in ``.git/config``'s
    ``remote.origin.url``.  This means the resulting clone has no
    embedded credentials and won't go stale when the agent's token
    is rotated.

    Returns a confirmation message with the local path.
    """
    resolved = _resolve_tool_path(local_path)

    clone_args = ["clone"]
    if bare:
        clone_args.append("--bare")
    clone_args.extend([remote_url, resolved])

    _, output = await _run_git(*clone_args, auth=True)
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

    Pushes the specified *branch_name* to *remote* (default
    ``origin``).  Credentials for HTTPS remotes belonging to the
    agent's project are injected per-call via subprocess environment
    variables (see :func:`_git_auth_env_for_current_agent`).
    """
    resolved = _resolve_tool_path(repo_path)
    _, output = await _run_git(
        "push", remote, branch_name, cwd=resolved, auth=True,
    )
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
    Credentials for the agent's project's forge are injected per-call
    (see :func:`_git_auth_env_for_current_agent`).
    """
    resolved = _resolve_tool_path(repo_path)
    _, output = await _run_git("fetch", remote, cwd=resolved, auth=True)
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
    fixup commits or base-branch updates).  Credentials for the
    agent's project's forge are injected per-call (see
    :func:`_git_auth_env_for_current_agent`).
    """
    resolved = _resolve_tool_path(repo_path)
    args = ["pull", remote]
    if branch is not None:
        args.append(branch)
    _, output = await _run_git(*args, cwd=resolved, auth=True)
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
