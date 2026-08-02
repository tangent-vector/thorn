"""Bootstrap an agency directory with a pre-configured ProjectCoordinator.

Creates the agent identity file (``<agent-id>.json``), the agent's
home directory, a ``MEMORY.md`` containing project-specific knowledge,
and a ``gateway.json`` service configuration (workspace, optional LLM
provider/model defaults, and project definition; forges are inferred
from the project URL).

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

The new on-disk shape (see :mod:`thorn.gateway._config`) lets the
project entry be just ``{name, url}`` for the common single-fork
case; the forge type, name, and API URL are inferred from the URL
host at startup, and the per-fork ``native_id`` and ``clone_url``
are parsed from the same URL.  Credentials are referenced by env
var name (``"env_var_name": "GITHUB_TOKEN"``); the literal value
is read from ``os.environ`` only at the points where it is needed
(broker registration, direct authentication).

Usage from code::

    from thorn.gateway._bootstrap import bootstrap_coordinator

    bootstrap_coordinator(
        agency_home=Path("/home/me/.thorn"),
        agency_workspace=Path("/home/me/thorn-workspace"),
        agent_id="thorn",
        project_name="example-repo",
        project_url="https://github.com/example-org/example-repo",
    )

GitHub uses PAT authentication (``$GITHUB_TOKEN`` by default).  GitHub
App installation tokens are not supported for the inferred event
source (the Notifications API requires user-scoped auth).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from thorn.core._provider import LLMConfig
from thorn.gateway._config import (
    GATEWAY_CONFIG_FILENAME,
    derive_forge_type_from_url,
)
from thorn.gateway._resources_helper import read_default_operator_agents_text
from thorn.runtime._paths import AgencyPaths
from thorn.runtime._session import AgentID

log = logging.getLogger(__name__)

CoordinatorAgentClassName = Literal[
    "ProjectCoordinator",
    "LeanProjectCoordinator",
]

_COORDINATOR_AGENT_CLASSES: set[str] = {
    "ProjectCoordinator",
    "LeanProjectCoordinator",
}


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
    project_entry: dict[str, Any],
    llm_config: LLMConfig | None = None,
) -> None:
    """Create or update ``gateway.json`` with workspace and project entry.

    The agency's workspace path is written as a top-level
    ``"workspace"`` string (overwriting any prior value).  The project
    entry is upserted by name so re-running the bootstrap for an
    additional project in the same agency accretes rather than
    clobbering.

    No ``forges:`` block is written.  The new gateway config infers
    the matching forge entry from the project URL at load time;
    operators only need to write an explicit ``forges:`` array when
    targeting a self-hosted forge whose type cannot be inferred.
    When ``llm_config`` is supplied, it replaces the gateway-level
    ``llm`` block; rerunning bootstrap without it preserves any existing
    LLM config.
    """
    config_path = agency_home / GATEWAY_CONFIG_FILENAME

    if config_path.is_file():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        data = {}

    data["workspace"] = workspace_path
    if llm_config is not None:
        data["llm"] = llm_config.model_dump(mode="json", exclude_none=True)

    projects: list[dict[str, Any]] = data.setdefault("projects", [])
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
    project_url: str,
    access_token_env: str | None = None,
    git_user_name: str = "",
    git_user_email: str = "",
    llm_config: LLMConfig | None = None,
    agent_class: CoordinatorAgentClassName = "ProjectCoordinator",
) -> AgentID:
    """Create a ProjectCoordinator agent in the given agency.

    *agency_home* is used as the agency home directory directly (no
    ``.thorn/`` subdirectory is appended).  The following files and
    directories are created under it:

    - ``<agency_home>/agents/<agent_id>/agent.json`` -- agent identity
    - ``<agency_home>/agents/<agent_id>/AGENTS.md`` -- operator-owned starter policy
    - ``<agency_home>/agents/<agent_id>/home/MEMORY.md`` -- agent memory
    - ``<agency_home>/gateway.json`` -- gateway service configuration

    *agency_workspace* is used as the agency's workspace root.  Its
    absolute path is written into ``gateway.json`` (top-level
    ``"workspace"`` field), and the per-agent session-workspace
    prefix ``<agency_workspace>/<agent_id>/`` is created eagerly so
    that misconfigured/unwritable workspaces fail at bootstrap rather
    than at first session.

    *project_url* is the human-facing URL of the project on its forge
    (e.g. ``"https://github.com/owner/repo"`` or
    ``"https://gitlab.com/group/project"``).  The forge entry is
    inferred from the URL host at load time; only the secret access
    token (a real secret) is referenced via ``$ENV_VAR`` in the agent
    identity file.

    The agent identity uses the new ``"accounts"`` shape: a flat
    list of account objects discriminated on ``service``.  GitHub
    accounts always use PAT authentication (default
    ``$GITHUB_TOKEN``) -- the inferred Notifications event source
    requires user-scoped credentials.

    *llm_config* records gateway-level LLM provider/model defaults.
    The config stores only non-secret values and an environment-variable
    name for the provider key; the literal key stays in the process
    environment.

    *agent_class* selects the persisted coordinator role.  The default
    ``"ProjectCoordinator"`` is the full production surface;
    ``"LeanProjectCoordinator"`` is an opt-in calibration role with a
    reduced prompt/tool surface for overhead experiments.

    Returns the ``AgentID`` of the created agent.
    """
    if agent_class not in _COORDINATOR_AGENT_CLASSES:
        choices = ", ".join(sorted(_COORDINATOR_AGENT_CLASSES))
        raise ValueError(
            f"Unknown coordinator agent class {agent_class!r}; "
            f"expected one of: {choices}."
        )

    forge_type = derive_forge_type_from_url(project_url)
    if forge_type is None:
        raise ValueError(
            f"Cannot infer forge type from project URL {project_url!r}: "
            "host is not a well-known forge.  Bootstrap currently only "
            "supports github.com and gitlab.com URLs; for self-hosted "
            "forges, write gateway.json by hand with an explicit "
            "`forges:` entry."
        )

    if access_token_env is None:
        access_token_env = _DEFAULT_TOKEN_ENV[forge_type]

    aid = AgentID(agent_id)

    # Resolve both roots up front so that the path written into
    # gateway.json is canonical and stable regardless of what the
    # caller passed (relative path, ``.``, symlink, etc.).
    agency_home = agency_home.resolve()
    agency_workspace = agency_workspace.resolve()

    paths = AgencyPaths.for_gateway(agency_home, agency_workspace)
    paths.agents_root.mkdir(parents=True, exist_ok=True)

    resolved_git_name = git_user_name or agent_id
    resolved_git_email = git_user_email or f"{agent_id}@thorn"

    # Build the credentials reference for the agent account.  The
    # account references its secret by env var name; broker
    # registration and direct-authentication paths read the literal
    # from ``os.environ`` at use time, so the agent's persisted
    # state never holds the secret.
    if forge_type == "github":
        credential_kind = "pat"
        service_name = "github"
    else:
        credential_kind = "gitlab-pat"
        service_name = "gitlab"
    credentials_list: list[dict[str, Any]] = [
        {"kind": credential_kind, "env_var_name": access_token_env},
    ]

    # -- Agent identity ------------------------------------------------------

    identity_path = paths.agent_identity_file(aid)
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    agent_data: dict[str, Any] = {
        "name": agent_id,
        "agent_class": agent_class,
        "metadata": {
            # ``metadata.project`` is intentionally retained for now;
            # killing it requires a follow-up redesign of git
            # identity resolution (see the deferred items in the
            # gateway-config-cleanup plan).
            "project": project_name,
        },
        "accounts": [
            {
                "service": service_name,
                "credentials": credentials_list,
                "git_user_name": resolved_git_name,
                "git_user_email": resolved_git_email,
            },
        ],
    }

    identity_path.write_text(
        json.dumps(agent_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote agent identity: %s", identity_path)

    # -- Operator-owned prompt policy ----------------------------------------
    #
    # The framework dir itself sits outside the agent's sandbox-visible
    # ``home/`` mount, so ``AGENTS.md`` here is the operator-owned slot
    # that phase-1 context gathering loads ahead of agent-home content.
    # Bootstrap writes a short, reviewable starter file rather than
    # burying role policy in Python defaults.

    operator_agents_path = paths.agent_framework_dir(aid) / "AGENTS.md"
    operator_agents_text = read_default_operator_agents_text().format(
        agent_id=agent_id,
        project_name=project_name,
    )
    operator_agents_path.write_text(operator_agents_text, encoding="utf-8")
    log.info("Wrote operator AGENTS.md: %s", operator_agents_path)

    # -- Agent home (memory) -------------------------------------------------
    #
    # The agent's home subtree is the ``home/`` directory inside its
    # framework dir; this is the part mounted into the tool-executor
    # sandbox, so MEMORY.md, journal/, etc. all go here.

    agent_home = paths.agent_home_mount(aid)
    agent_home.mkdir(parents=True, exist_ok=True)

    memory_path = agent_home / "MEMORY.md"
    memory_lines = [
        f"# {project_name} Coordinator Memory",
        "",
        f"- **Project service**: `{project_name}`",
        f"- **Project URL**: {project_url}",
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
    # Sessions create their own
    # ``<agent_workspace_mount>/<session_key>/`` subdirectories
    # lazily, but creating the agent-level workspace (and the sibling
    # control dir the tool-host socket lives in) here means any
    # "workspace path is unwritable" error surfaces during bootstrap
    # rather than the first time a session tries to start or the
    # daemon tries to listen.

    agent_workspace = paths.agent_workspace_mount(aid)
    agent_workspace.mkdir(parents=True, exist_ok=True)
    paths.agent_control_dir(aid).mkdir(parents=True, exist_ok=True)
    log.info("Created agent workspace: %s", agent_workspace)

    # -- Gateway configuration -----------------------------------------------

    project_entry: dict[str, Any] = {
        "name": project_name,
        "url": project_url,
    }

    _ensure_gateway_config(
        agency_home,
        workspace_path=str(agency_workspace),
        project_entry=project_entry,
        llm_config=llm_config,
    )

    return aid


__all__ = [
    "CoordinatorAgentClassName",
    "bootstrap_coordinator",
]
