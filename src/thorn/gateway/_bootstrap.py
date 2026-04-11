"""Bootstrap a Runtime directory with a pre-configured ProjectCoordinator.

Creates the agent identity file (``<agent-id>.json``), workspace
directory, a ``MEMORY.md`` containing project-specific knowledge, and
a ``gateway.json`` service configuration.  The result is a Runtime
directory ready for ``thorn serve``.

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
from typing import Any

from thorn.gateway._config import GATEWAY_CONFIG_FILENAME
from thorn.runtime._session import AgentID

log = logging.getLogger(__name__)


def _ensure_gateway_config(
    thorn_dir: Path,
    *,
    service_name: str,
    source_type: str,
    source_config: dict[str, Any],
) -> None:
    """Create or update ``gateway.json`` with a service entry.

    If the file already exists, the new entry is appended unless an
    entry with the same ``name`` is already present (in which case
    it is updated in place).  If the file does not exist, it is
    created.
    """
    config_path = thorn_dir / GATEWAY_CONFIG_FILENAME

    if config_path.is_file():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        data = {"services": []}

    services: list[dict[str, Any]] = data.setdefault("services", [])

    new_entry = {
        "name": service_name,
        "type": source_type,
        "config": source_config,
    }

    for i, existing in enumerate(services):
        if existing.get("name") == service_name:
            services[i] = new_entry
            break
    else:
        services.append(new_entry)

    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote gateway config: %s", config_path)


def bootstrap_coordinator(
    *,
    runtime_root: Path,
    agent_id: str,
    project_name: str,
    clone_url: str,
    default_branch: str = "main",
    project_id: int | None = None,
    access_token_env: str = "GITLAB_TOKEN",
    gitlab_url_env: str = "GITLAB_URL",
) -> AgentID:
    """Create a ProjectCoordinator agent in the given Runtime directory.

    Creates the directory structure expected by ``SessionStore`` and
    the gateway configuration file:

    - ``<runtime_root>/.thorn/agents/<agent_id>.json``
    - ``<runtime_root>/.thorn/agents/<agent_id>/MEMORY.md``
    - ``<runtime_root>/.thorn/gateway.json``

    Returns the ``AgentID`` of the created agent.
    """
    aid = AgentID(agent_id)
    thorn_dir = runtime_root / ".thorn"
    agents_root = thorn_dir / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)

    # -- Agent identity ------------------------------------------------------

    identity_path = agents_root / f"{agent_id}.json"
    agent_data: dict[str, Any] = {
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

    # -- Agent workspace & memory --------------------------------------------

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
        "## Active work",
        "",
        "_No active issues or MRs yet._",
    ])

    memory_path.write_text("\n".join(memory_lines) + "\n", encoding="utf-8")
    log.info("Wrote agent memory: %s", memory_path)

    # -- Gateway service configuration ---------------------------------------

    _ensure_gateway_config(
        thorn_dir,
        service_name=agent_id,
        source_type="gitlab",
        source_config={
            "url": f"${gitlab_url_env}",
            "token": f"${access_token_env}",
        },
    )

    return aid


__all__ = [
    "bootstrap_coordinator",
]
