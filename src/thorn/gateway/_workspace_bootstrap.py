"""Deterministic repository checkout bootstrap for gateway sessions."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from thorn._redaction import redact_secret_snippet

_GIT_COMMAND_TIMEOUT_S = 120.0
_MAX_DIAGNOSTIC_CHARS = 4000


class RepositoryCloneURL(str):
    """Clone location for a repository checkout."""

    def __new__(cls, value: str) -> "RepositoryCloneURL":
        stripped = value.strip()
        if not stripped:
            raise ValueError("Repository clone URL must be non-empty")
        return str.__new__(cls, stripped)


class GitBranchName(str):
    """Git branch name used for an initial checkout."""

    def __new__(cls, value: str) -> "GitBranchName":
        stripped = value.strip()
        if not stripped:
            raise ValueError("Git branch name must be non-empty")
        if "\0" in stripped:
            raise ValueError("Git branch name must not contain NUL")
        return str.__new__(cls, stripped)


@dataclass(frozen=True)
class RepositoryCheckoutSpec:
    """Repository checkout that should exist before a session prompt."""

    clone_url: RepositoryCloneURL
    default_branch: GitBranchName | None = None

    @classmethod
    def from_event_fields(
        cls,
        *,
        clone_url: str | None,
        default_branch: str | None = "",
    ) -> "RepositoryCheckoutSpec | None":
        """Build a checkout spec from forge event fields, if possible."""
        if clone_url is None or not clone_url.strip():
            return None
        branch = (
            GitBranchName(default_branch)
            if default_branch is not None and default_branch.strip()
            else None
        )
        return cls(
            clone_url=RepositoryCloneURL(clone_url),
            default_branch=branch,
        )


class WorkspaceBootstrapError(RuntimeError):
    """Raised when Thorn cannot prepare a session workspace checkout."""


async def bootstrap_repository_checkout(
    *,
    workspace: Path,
    checkout: RepositoryCheckoutSpec,
) -> None:
    """Ensure *workspace* contains the repository described by *checkout*.

    Fresh or missing workspaces are cloned with the repository at the
    workspace root.  Existing git checkouts are fetched in place.  A
    non-empty, non-git directory is treated as a hard configuration
    problem so Thorn does not prompt the agent with unrelated local
    files masquerading as project context.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    if _contains_git_checkout(workspace):
        await _run_git(("fetch", "--prune", "origin"), cwd=workspace)
        return

    if not _is_empty_directory(workspace):
        entries = ", ".join(sorted(path.name for path in workspace.iterdir())[:5])
        raise WorkspaceBootstrapError(
            "Cannot bootstrap repository checkout into non-empty "
            f"workspace without .git: {workspace} ({entries})"
        )

    clone_args: list[str] = ["clone"]
    if checkout.default_branch is not None:
        clone_args.extend(["--branch", str(checkout.default_branch)])
    clone_args.extend([str(checkout.clone_url), "."])
    await _run_git(tuple(clone_args), cwd=workspace)


def _contains_git_checkout(workspace: Path) -> bool:
    return (workspace / ".git").exists()


def _is_empty_directory(workspace: Path) -> bool:
    return next(workspace.iterdir(), None) is None


async def _run_git(args: Sequence[str], *, cwd: Path) -> None:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=_GIT_COMMAND_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        command = _redacted_git_command(args)
        raise WorkspaceBootstrapError(
            f"Timed out while running workspace bootstrap command: {command}"
        ) from exc

    output = stdout.decode(errors="replace") if stdout else ""
    if proc.returncode == 0:
        return

    command = _redacted_git_command(args)
    snippet = redact_secret_snippet(
        output,
        max_chars=_MAX_DIAGNOSTIC_CHARS,
    )
    raise WorkspaceBootstrapError(
        "Workspace bootstrap command failed "
        f"(exit={proc.returncode}): {command}\n{snippet}"
    )


def _redacted_git_command(args: Sequence[str]) -> str:
    return redact_secret_snippet(
        " ".join(("git", *args)),
        max_chars=_MAX_DIAGNOSTIC_CHARS,
    )


__all__ = [
    "GitBranchName",
    "RepositoryCheckoutSpec",
    "RepositoryCloneURL",
    "WorkspaceBootstrapError",
    "bootstrap_repository_checkout",
]
