"""Bootstrap a Runtime directory with a pre-configured ProjectCoordinator.

Creates the agent identity file (``<agent-id>.json``), workspace
directory, a ``MEMORY.md`` containing project-specific knowledge, and
a ``gateway.json`` service configuration (forge definition and project
definition -- no event sources, as those are inferred at startup).

The agent identity file includes an ``"accounts"`` section with forge
credentials (using ``$ENV_VAR`` references for secrets) and literal
git identity fields.

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
        forge_base_url="https://gitlab-master.nvidia.com/api/v4",
    )

    bootstrap_coordinator(
        runtime_root=Path("my-runtime"),
        agent_id="gh-coordinator",
        project_name="my-repo",
        clone_url="https://github.com/owner/repo.git",
        native_project_id="owner/repo",
        forge_type="github",
        forge_base_url="https://api.github.com",
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


def _upsert_by_name(
    items: list[dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    """Insert or replace an entry by ``name`` field."""
    name = entry["name"]
    for i, existing in enumerate(items):
        if existing.get("name") == name:
            items[i] = entry
            return
    items.append(entry)


def _ensure_gateway_config(
    thorn_dir: Path,
    *,
    forge_entry: dict[str, Any],
    project_entry: dict[str, Any],
) -> None:
    """Create or update ``gateway.json`` with forge and project entries.

    Uses the new format with top-level ``"forges"`` and ``"projects"``
    arrays.  Each entry is upserted by name.
    """
    config_path = thorn_dir / GATEWAY_CONFIG_FILENAME

    if config_path.is_file():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        data = {}

    forges: list[dict[str, Any]] = data.setdefault("forges", [])
    projects: list[dict[str, Any]] = data.setdefault("projects", [])

    _upsert_by_name(forges, forge_entry)
    _upsert_by_name(projects, project_entry)

    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote gateway config: %s", config_path)


def _build_github_auth_block(
    auth_mode: str,
    token_env: str,
) -> dict[str, str]:
    """Build the ``credentials`` sub-object for a GitHub agent account.

    *auth_mode* is ``"pat"`` (default -- uses ``$<token_env>``) or
    ``"app"`` (uses ``$GITHUB_APP_*`` env var references).
    """
    if auth_mode == "app":
        return {
            "kind": "app",
            "app_id": "$GITHUB_APP_ID",
            "installation_id": "$GITHUB_APP_INSTALLATION_ID",
            "private_key_pem": "$GITHUB_APP_PRIVATE_KEY",
        }
    return {
        "kind": "pat",
        "token": f"${token_env}",
    }


_FORGE_DEFAULTS: dict[str, tuple[str, str]] = {
    "gitlab": ("GITLAB_TOKEN", "GITLAB_URL"),
    "github": ("GITHUB_TOKEN", "GITHUB_API_URL"),
}


def bootstrap_coordinator(
    *,
    runtime_root: Path,
    agent_id: str,
    project_name: str,
    clone_url: str,
    default_branch: str = "main",
    native_project_id: str = "",
    forge_type: str = "gitlab",
    access_token_env: str | None = None,
    forge_base_url: str = "",
    forge_service_name: str = "",
    git_user_name: str = "",
    git_user_email: str = "",
    github_auth_mode: str = "pat",
    # Legacy parameters (accepted but mapped to new fields)
    project_id: int | None = None,
    forge_url_env: str | None = None,
    gitlab_url_env: str = "",
) -> AgentID:
    """Create a ProjectCoordinator agent in the given Runtime directory.

    Creates the directory structure expected by ``SessionStore`` and
    the gateway configuration file:

    - ``<runtime_root>/.thorn/agents/<agent_id>.json``
    - ``<runtime_root>/.thorn/agents/<agent_id>/MEMORY.md``
    - ``<runtime_root>/.thorn/gateway.json``

    The gateway config includes a forge definition and a project
    definition.  Event sources are inferred at startup from the
    agent's account on the forge (no explicit event source entry).

    The agent identity includes an ``"accounts"`` section with forge
    credentials (``$ENV_VAR`` for secrets) and literal git identity.
    ``metadata.project`` is also set for backward compatibility with
    code that reads it.

    For GitHub, *github_auth_mode* selects between ``"pat"``
    (``$GITHUB_TOKEN``, the default) and ``"app"``
    (``$GITHUB_APP_*``).

    Returns the ``AgentID`` of the created agent.
    """
    if gitlab_url_env:
        forge_url_env = gitlab_url_env
    if project_id is not None and not native_project_id:
        native_project_id = str(project_id)
    if not forge_service_name:
        forge_service_name = f"{project_name}-forge"

    default_token, _default_url_env = _FORGE_DEFAULTS.get(
        forge_type, ("GITLAB_TOKEN", "GITLAB_URL"),
    )
    if access_token_env is None:
        access_token_env = default_token

    aid = AgentID(agent_id)
    thorn_dir = runtime_root / ".thorn"
    agents_root = thorn_dir / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)

    resolved_git_name = git_user_name or agent_id
    resolved_git_email = git_user_email or f"{agent_id}@thorn"

    # -- Build credentials for the agent account ----------------------------

    if forge_type == "github":
        credentials_block: dict[str, Any] = _build_github_auth_block(
            github_auth_mode, access_token_env,
        )
    else:
        credentials_block = {
            "kind": "gitlab-pat",
            "token": f"${access_token_env}",
        }

    # -- Agent identity ------------------------------------------------------

    identity_path = agents_root / f"{agent_id}.json"
    agent_data: dict[str, Any] = {
        "id": str(aid),
        "agent_class": "ProjectCoordinator",
        "name": agent_id,
        "metadata": {
            "project": project_name,
        },
        "accounts": {
            "forge_accounts": [
                {
                    "forge": forge_service_name,
                    "credentials": credentials_block,
                    "git_user_name": resolved_git_name,
                    "git_user_email": resolved_git_email,
                },
            ],
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

    # -- Gateway configuration (new format) ----------------------------------

    forge_entry: dict[str, Any] = {
        "name": forge_service_name,
        "type": forge_type,
    }
    if forge_base_url:
        forge_entry["base_url"] = forge_base_url

    project_entry: dict[str, Any] = {
        "name": project_name,
        "default_branch": default_branch,
        "forks": [
            {
                "forge": forge_service_name,
                "native_id": native_project_id,
                "name": "upstream",
                "clone_url": clone_url,
            },
        ],
    }

    _ensure_gateway_config(
        thorn_dir,
        forge_entry=forge_entry,
        project_entry=project_entry,
    )

    return aid


__all__ = [
    "bootstrap_coordinator",
]
