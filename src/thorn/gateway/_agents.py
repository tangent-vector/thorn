"""Agent subclasses for gateway operation.

Defines the ``ProjectCoordinator`` agent role: a persistent agent
responsible for a single GitLab project.  It processes incoming events
(e.g. @-mentions, assignments) and performs the necessary actions:
reading issues, cloning repositories, creating branches, making changes,
pushing, and opening merge requests.

For the initial vertical slice, the coordinator handles coding tasks
directly (single-agent shortcut) rather than delegating to a developer
sub-agent.
"""

from __future__ import annotations

from typing import Any, ClassVar

from thorn.core._agent import Agent
from thorn.core._tools import FILE_READING, FILE_WRITING
from thorn.tools.git import GIT_TOOLS
from thorn.tools.gitlab import GITLAB_TOOLS


_COORDINATOR_SYSTEM_PROMPT = """\
You are a project coordinator agent managing a GitLab project.

Your responsibilities:
- Read and understand incoming GitLab notifications (issues, comments, MR reviews).
- When asked to make code changes: clone the project repository, create a working \
branch, make the requested changes, commit, push, and open a merge request.
- Post a comment on the original issue or MR linking to any MR you create.
- Mark the GitLab TODO as done once you have fully handled the notification.

Workflow for code changes:
1. Read the issue/comment to understand what is being requested.
2. Clone the repository using git_clone (bare clone) if you haven't already. \
   Use the clone URL from the notification or from your memory.
3. Create a worktree with git_worktree_add for a new branch \
   (e.g. thorn/issue-<iid>), branching from the default branch.
4. Read relevant files, make changes using edit_file or create_file.
5. Commit your changes with git_commit.
6. Push the branch with git_push.
7. Create a merge request with create_merge_request.
8. Post a comment on the original issue linking to the MR.
9. Mark the TODO as done with gitlab_mark_todo_done.

Important:
- Paths for file operations are relative to the worktree directory \
  (which is your workspace after you set it up).
- The git clone URL should be taken from the notification metadata or \
  your MEMORY.md.
- Credentials for git operations are handled transparently — just use \
  the URL as provided.
- Keep commit messages and MR descriptions clear and concise.
"""


class ProjectCoordinator(Agent):
    """Persistent agent responsible for managing a single GitLab project.

    Combines GitLab API tools, git tools, and file I/O tools so that it
    can process incoming events end-to-end: from reading an issue to
    opening a merge request.

    For the vertical slice, the coordinator handles coding work directly.
    In the future, it will delegate to ``DeveloperAgent`` sub-agents via
    a ``delegate_task`` tool.
    """

    system_prompts: ClassVar[list[Any]] = [_COORDINATOR_SYSTEM_PROMPT]
    tools: ClassVar[list[Any]] = [GITLAB_TOOLS, GIT_TOOLS, FILE_READING, FILE_WRITING]


__all__ = [
    "ProjectCoordinator",
]
