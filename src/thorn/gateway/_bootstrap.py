"""Bootstrap a Runtime directory with a pre-configured ProjectCoordinator.

Creates the agent identity file (``<agent-id>.json``), workspace
directory, a ``MEMORY.md`` containing project-specific knowledge, and
a ``gateway.json`` service configuration (forge service + project
service).  The result is a Runtime directory ready for ``thorn serve``.

Usage from code::

    from thorn.gateway._bootstrap import bootstrap_coordinator

    bootstrap_coordinator(
        runtime_root=Path("my-runtime"),
        agent_id="lace-coordinator",
        project_name="lace",
        clone_url="https://gitlab-master.nvidia.com/lace/lace.git",
        default_branch="main",
        native_project_id="214768",
        forge_type="gitlab",
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


def _upsert_service(
    services: list[dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    """Insert or replace a service entry by name."""
    name = entry["name"]
    for i, existing in enumerate(services):
        if existing.get("name") == name:
            services[i] = entry
            return
    services.append(entry)


def _ensure_gateway_config(
    thorn_dir: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Create or update ``gateway.json`` with multiple service entries.

    Each entry is upserted by name — existing entries with the same
    name are replaced, new entries are appended.
    """
    config_path = thorn_dir / GATEWAY_CONFIG_FILENAME

    if config_path.is_file():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        data = {"services": []}

    services: list[dict[str, Any]] = data.setdefault("services", [])

    for entry in entries:
        _upsert_service(services, entry)

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
    native_project_id: str = "",
    forge_type: str = "gitlab",
    access_token_env: str = "GITLAB_TOKEN",
    forge_url_env: str = "GITLAB_URL",
    forge_service_name: str = "",
    # Legacy parameters (accepted but mapped to new fields)
    project_id: int | None = None,
    gitlab_url_env: str = "",
) -> AgentID:
    """Create a ProjectCoordinator agent in the given Runtime directory.

    Creates the directory structure expected by ``SessionStore`` and
    the gateway configuration file:

    - ``<runtime_root>/.thorn/agents/<agent_id>.json``
    - ``<runtime_root>/.thorn/agents/<agent_id>/MEMORY.md``
    - ``<runtime_root>/.thorn/gateway.json``

    The gateway config includes a forge service and a project service.
    The agent identity references the project service by name.

    Returns the ``AgentID`` of the created agent.
    """
    if gitlab_url_env:
        forge_url_env = gitlab_url_env
    if project_id is not None and not native_project_id:
        native_project_id = str(project_id)
    if not forge_service_name:
        forge_service_name = f"{project_name}-forge"

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
            "project": project_name,
        },
    }

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
        f"- **Project service**: `{project_name}`",
        f"- **Clone URL**: {clone_url}",
        f"- **Default branch**: {default_branch}",
    ]

    memory_lines.extend([
        "",
        "## Active work",
        "",
        "_No active issues or change requests yet._",
    ])

    memory_path.write_text("\n".join(memory_lines) + "\n", encoding="utf-8")
    log.info("Wrote agent memory: %s", memory_path)

    # -- Gateway service configuration ---------------------------------------

    forge_entry = {
        "name": forge_service_name,
        "type": forge_type,
        "config": {
            "url": f"${forge_url_env}",
            "token": f"${access_token_env}",
        },
    }

    project_entry: dict[str, Any] = {
        "name": project_name,
        "type": "project",
        "config": {
            "forge": forge_service_name,
            "native_id": native_project_id,
            "path": project_name,
            "clone_url": clone_url,
            "default_branch": default_branch,
        },
    }

    _ensure_gateway_config(thorn_dir, [forge_entry, project_entry])

    return aid


__all__ = [
    "bootstrap_coordinator",
]
