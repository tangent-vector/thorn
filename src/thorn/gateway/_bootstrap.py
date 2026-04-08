"""Bootstrap a Runtime directory with a pre-configured ProjectCoordinator.

Creates the agent identity file (``<agent-id>.json``), workspace
directory, and a ``MEMORY.md`` containing project-specific knowledge.
The result is a Runtime directory ready for ``thorn serve``.

Usage from code::

    from thorn.gateway._bootstrap import bootstrap_coordinator

    bootstrap_coordinator(
        runtime_root=Path("my-runtime"),
        agent_id="lace-coordinator",
        project_name="lace-lang",
        clone_url="https://gitlab.example.com/group/lace-lang.git",
        default_branch="main",
        project_id=214768,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from thorn.runtime._session import AgentID

log = logging.getLogger(__name__)


def bootstrap_coordinator(
    *,
    runtime_root: Path,
    agent_id: str,
    project_name: str,
    clone_url: str,
    default_branch: str = "main",
    project_id: int | None = None,
    access_token_env: str = "GITLAB_TOKEN",
) -> AgentID:
    """Create a ProjectCoordinator agent in the given Runtime directory.

    Creates the directory structure expected by ``SessionStore``:

    - ``<runtime_root>/.thorn/agents/<agent_id>.json``
    - ``<runtime_root>/.thorn/agents/<agent_id>/MEMORY.md``

    Returns the ``AgentID`` of the created agent.
    """
    aid = AgentID(agent_id)
    agents_root = runtime_root / ".thorn" / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)

    identity_path = agents_root / f"{agent_id}.json"
    agent_data = {
        "id": str(aid),
        "agent_class": "ProjectCoordinator",
        "name": agent_id,
        "metadata": {
            "access_token": f"${access_token_env}",
            "project_name": project_name,
            "clone_url": clone_url,
            "default_branch": default_branch,
        },
    }
    if project_id is not None:
        agent_data["metadata"]["project_id"] = project_id

    identity_path.write_text(
        json.dumps(agent_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote agent identity: %s", identity_path)

    workspace_dir = agents_root / agent_id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    memory_path = workspace_dir / "MEMORY.md"
    memory_lines = [
        f"# {project_name} Coordinator Memory",
        "",
        f"- **Project name**: {project_name}",
        f"- **Clone URL**: {clone_url}",
        f"- **Default branch**: {default_branch}",
    ]
    if project_id is not None:
        memory_lines.append(f"- **Project ID**: {project_id}")

    memory_lines.extend([
        "",
        "## Conventions",
        "",
        "- Branch naming: `thorn/issue-<iid>`",
        "- Bare clone location: `repos/<project-name>/`",
        "- Worktrees: `worktrees/thorn/issue-<iid>/`",
    ])

    memory_path.write_text("\n".join(memory_lines) + "\n", encoding="utf-8")
    log.info("Wrote agent memory: %s", memory_path)

    return aid


__all__ = [
    "bootstrap_coordinator",
]
