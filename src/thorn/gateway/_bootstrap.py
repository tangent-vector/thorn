"""Bootstrap an agency directory with a pre-configured ProjectCoordinator.

Creates the agent identity file (``<agent-id>.json``), the agent's
home directory, a ``MEMORY.md`` containing project-specific knowledge,
and a ``gateway.json`` service configuration (forge definition,
project definition, and the agency's workspace path -- no event
sources, as those are inferred at startup).

The bootstrap takes two filesystem roots, matching the
home-vs-workspace split that ``AgencyPaths`` already exposes for the
gateway runtime:

- ``agency_home``: the directory used as-is for the agency's persisted
  state (identities, sessions, memory, journals, service queues, and
  ``gateway.json`` itself).  Nothing is nested under a ``.thorn/``
  subdirectory; whatever path is passed in *is* the agency home.

- ``agency_workspace``: the directory used as the agency's workspace
  root, where agent sessions do their work.  The absolute path is
  written into ``gateway.json`` so that ``thorn serve`` can locate it
  later without having to ask for the workspace on every startup.

All non-secret values (forge ``base_url``, project metadata, git
identity, agency workspace path) are written *literally* into the
on-disk JSON.  Only the secret access token uses an ``$ENV_VAR``
reference, in line with the "config in JSON, secrets via env" model.

Usage from code::

    from thorn.gateway._bootstrap import bootstrap_coordinator

    bootstrap_coordinator(
        agency_home=Path("/home/me/.thorn"),
        agency_workspace=Path("/home/me/thorn-workspace"),
        agent_id="lace-coordinator",
        project_name="lace",
        clone_url="https://gitlab-master.nvidia.com/lace/lace.git",
        default_branch="main",
        native_project_id="214768",
        forge_type="gitlab",
        forge_base_url="https://gitlab-master.nvidia.com/api/v4",
    )

    bootstrap_coordinator(
        agency_home=Path("/home/me/.thorn"),
        agency_workspace=Path("/home/me/thorn-workspace"),
        agent_id="gh-coordinator",
        project_name="my-repo",
        clone_url="https://github.com/owner/repo.git",
        native_project_id="owner/repo",
        forge_type="github",
        forge_base_url="https://api.github.com",
    )

GitHub uses PAT authentication (``$GITHUB_TOKEN`` by default).  GitHub
App installation tokens are not supported for the inferred event
source (the Notifications API requires user-scoped auth).
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
    agency_home: Path,
    *,
    workspace_path: str,
    forge_entry: dict[str, Any],
    project_entry: dict[str, Any],
) -> None:
    """Create or update ``gateway.json`` with workspace, forge, and project entries.

    The agency's workspace path is written as a top-level
    ``"workspace"`` string (overwriting any prior value).  Forge and
    project entries are upserted by name so that re-running the
    bootstrap for additional coordinators in the same agency accretes
    rather than clobbering.
    """
    config_path = agency_home / GATEWAY_CONFIG_FILENAME

    if config_path.is_file():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        data = {}

    # Always write the workspace path; the bootstrap is the canonical
    # place to set it, and a later bootstrap call against the same
    # agency home should normally be using the same workspace anyway.
    data["workspace"] = workspace_path

    forges: list[dict[str, Any]] = data.setdefault("forges", [])
    projects: list[dict[str, Any]] = data.setdefault("projects", [])

    _upsert_by_name(forges, forge_entry)
    _upsert_by_name(projects, project_entry)

    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote gateway config: %s", config_path)


_DEFAULT_TOKEN_ENV: dict[str, str] = {
    "gitlab": "GITLAB_TOKEN",
    "github": "GITHUB_TOKEN",
}


def bootstrap_coordinator(
    *,
    agency_home: Path,
    agency_workspace: Path,
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
) -> AgentID:
    """Create a ProjectCoordinator agent in the given agency.

    *agency_home* is used as the agency home directory directly (no
    ``.thorn/`` subdirectory is appended).  The following files and
    directories are created under it:

    - ``<agency_home>/agents/<agent_id>.json`` -- agent identity
    - ``<agency_home>/agents/<agent_id>/MEMORY.md`` -- agent memory
    - ``<agency_home>/gateway.json`` -- gateway service configuration

    *agency_workspace* is used as the agency's workspace root.  Its
    absolute path is written into ``gateway.json`` (top-level
    ``"workspace"`` field), and the per-agent session-workspace
    prefix ``<agency_workspace>/<agent_id>/`` is created eagerly so
    that misconfigured/unwritable workspaces fail at bootstrap rather
    than at first session.

    *forge_base_url* is written literally into the gateway forge entry.
    The gateway then resolves the base URL from JSON at startup; no
    environment variable lookup is performed for the URL.  Only the
    access token (a real secret) is referenced via ``$ENV_VAR`` in the
    agent identity file.  Event sources are inferred at startup from
    the agent's account on the forge — no explicit event source entry
    is written.

    The agent identity includes an ``"accounts"`` section with forge
    credentials (``$ENV_VAR`` for the secret token) and literal git
    identity.  ``metadata.project`` is also set for backward
    compatibility with code that reads it.

    GitHub always uses PAT authentication (default ``$GITHUB_TOKEN``).

    Returns the ``AgentID`` of the created agent.
    """
    if not forge_service_name:
        forge_service_name = f"{project_name}-forge"

    if access_token_env is None:
        access_token_env = _DEFAULT_TOKEN_ENV.get(forge_type, "GITLAB_TOKEN")

    aid = AgentID(agent_id)

    # Resolve both roots up front so that the path written into
    # gateway.json is canonical and stable regardless of what the
    # caller passed (relative path, ``.``, symlink, etc.).
    agency_home = agency_home.resolve()
    agency_workspace = agency_workspace.resolve()

    agents_root = agency_home / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)

    resolved_git_name = git_user_name or agent_id
    resolved_git_email = git_user_email or f"{agent_id}@thorn"

    # -- Build credentials for the agent account ----------------------------

    if forge_type == "github":
        credentials_block: dict[str, Any] = {
            "kind": "pat",
            "token": f"${access_token_env}",
        }
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

    # -- Agent home (memory) -------------------------------------------------

    agent_home = agents_root / agent_id
    agent_home.mkdir(parents=True, exist_ok=True)

    memory_path = agent_home / "MEMORY.md"
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

    # -- Per-agent workspace prefix ------------------------------------------
    #
    # Sessions create their own ``<agent_workspace>/<session_key>/``
    # subdirectories lazily, but creating the agent-level prefix here
    # means any "workspace path is unwritable" error surfaces during
    # bootstrap rather than the first time a session tries to start.

    agent_workspace = agency_workspace / agent_id
    agent_workspace.mkdir(parents=True, exist_ok=True)
    log.info("Created agent workspace: %s", agent_workspace)

    # -- Gateway configuration -----------------------------------------------

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
        agency_home,
        workspace_path=str(agency_workspace),
        forge_entry=forge_entry,
        project_entry=project_entry,
    )

    return aid


__all__ = [
    "bootstrap_coordinator",
]
