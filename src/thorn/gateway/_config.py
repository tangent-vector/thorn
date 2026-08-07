"""Gateway configuration: loading services from an agency config file.

The agency configuration file lives in the *agency home* directory and
declares the agency's workspace directory along with its forges and
projects.  The preferred filename is ``agency.yaml``; ``agency.json``,
``gateway.yaml``, and ``gateway.json`` are also accepted.  Forges are
external platforms that host version-controlled repositories; projects
are logical software projects with one or more forks hosted on those
forges.

The on-disk format uses a top-level ``"workspace"`` string and typed
arrays.  In the simplest case, the user only writes ``"projects"``
and the rest is inferred::

    {
      "workspace": "/home/me/thorn-workspace",
      "projects": [
        {
          "name": "tiny-talk",
          "url": "https://github.com/example-org/example-repo"
        }
      ]
    }

The forge entry for ``github.com`` is synthesized from the project's
URL.  An explicit ``"forges"`` block is only required when (a) the
forge host is not one of the well-known ones (``github.com``,
``gitlab.com``) so its type cannot be inferred, or (b) the user wants
to override defaults like ``poll_interval``.

A multi-fork project specifies its forks explicitly::

    {
      "name": "lace",
      "forks": [
        { "url": "https://gitlab.example.com/lace/lace" },
        { "url": "https://github.com/me/lace-fork", "name": "fork" }
      ]
    }

The ``"workspace"`` value identifies the agency's *workspace root*:
where agent sessions do their work (clone repositories, edit files,
run builds).  It is independent of the agency home directory that
holds this config file.  An absolute path is used as-is; a relative
path is resolved against the agency home -- see
:meth:`GatewayConfig.resolve_workspace`.

Future plug-in service categories will be added as additional typed
arrays alongside ``forges:`` and ``projects:`` (for example, a
``messaging_services:`` array).  Heterogeneous arrays -- those whose
entries can be one of several backends keyed by ``"type"`` -- are
instantiated through :class:`ServiceTypeRegistry`.

Event sources are **not** configured explicitly.  They are inferred
at startup from agent accounts on registered forges (see
:func:`infer_event_sources`).

String values that begin with ``$`` are treated as environment
variable references and expanded at load time, keeping secrets out of
the config file itself.  Per the design convention, only genuinely
secret values (tokens, private keys) should use ``$ENV_VAR``; all
other configuration should be literal.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema, core_schema

from thorn.core._provider import LLMConfig
from thorn.core._service import Service
from thorn.gateway._event import EventSource
from thorn.gateway._peer import PeerSpec
from thorn.gateway._trigger_policy import UnknownActorPolicyMode
from thorn.runtime._session import AgentID

log = logging.getLogger(__name__)

GATEWAY_CONFIG_FILENAME = "gateway.json"
AGENCY_CONFIG_FILENAMES = (
    "agency.yaml",
    "agency.json",
    "gateway.yaml",
    GATEWAY_CONFIG_FILENAME,
)


class AgencyConfigFileFormat(StrEnum):
    """Supported on-disk formats for agency configuration files."""

    JSON = "json"
    YAML = "yaml"


@dataclass(frozen=True)
class AgencyConfigFile:
    """One discovered agency configuration file."""

    path: Path
    configuration_format: AgencyConfigFileFormat


class PeerAccountIDPolicy(StrEnum):
    """How config resolution treats handle-only peer accounts."""

    REJECT_RESOLVABLE_HANDLES = "reject_resolvable_handles"
    ALLOW_HANDLE_ONLY = "allow_handle_only"


@dataclass(frozen=True)
class PeerAccountHandleOnlyProblem:
    """A peer account that needs immutable-ID resolution before startup."""

    peer_id: str
    service_name: str
    account_id: str
    forge_type: str


_FORGE_TYPES_WITH_IMMUTABLE_USER_IDS = frozenset({"github", "gitlab"})


# ---------------------------------------------------------------------------
# Forge URL <-> type/name inference helpers
# ---------------------------------------------------------------------------

# Hosts whose forge type can be inferred without operator input.  We
# only include the canonical SaaS hosts here; any self-hosted forge
# (GitHub Enterprise, self-hosted GitLab) must be declared explicitly
# in the ``"forges"`` block so the operator's intent is unambiguous.
_KNOWN_FORGE_HOSTS_BY_TYPE: dict[str, str] = {
    "github.com": "github",
    "gitlab.com": "gitlab",
}


def _host_of(url: str) -> str:
    """Return the lowercase hostname of *url* (no port, no userinfo)."""
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


def derive_forge_type_from_url(url: str) -> str | None:
    """Infer the forge backend type from a forge or fork URL.

    Returns ``"github"`` or ``"gitlab"`` for the well-known hosts
    (``github.com``, ``gitlab.com``).  Returns ``None`` for any other
    host so that the caller can require an explicit ``ForgeSpec.type``.
    """
    host = _host_of(url)
    if not host:
        return None
    if host.startswith("api."):
        host = host[4:]
    return _KNOWN_FORGE_HOSTS_BY_TYPE.get(host)


def derive_forge_name_from_url(url: str) -> str:
    """Derive a default forge name from a forge URL.

    Strips a leading ``api.`` and the public TLD for the well-known
    SaaS hosts so that ``https://github.com`` becomes ``"github"`` and
    ``https://gitlab.com`` becomes ``"gitlab"``.  Other hosts get the
    full hostname with dots replaced by hyphens, e.g.
    ``"gitlab-internal-example-com"``.

    Raises :class:`ValueError` when *url* has no hostname.
    """
    host = _host_of(url)
    if not host:
        raise ValueError(
            f"Cannot derive a forge name from URL {url!r}: no hostname."
        )
    if host.startswith("api."):
        host = host[4:]
    if host in _KNOWN_FORGE_HOSTS_BY_TYPE:
        # ``github.com`` -> ``github``; ``gitlab.com`` -> ``gitlab``.
        return host.split(".")[0]
    return host.replace(".", "-")


def derive_api_url(forge_type: str, url: str) -> str:
    """Derive the API base URL for a forge given its type and human URL.

    For ``github`` on ``github.com`` returns ``"https://api.github.com"``;
    for GitHub Enterprise (any other host) returns
    ``"https://<host>/api/v3"`` per the GHE convention.

    For ``gitlab`` returns the instance URL itself: python-gitlab takes
    the instance URL (e.g. ``"https://gitlab.com"``) and appends
    ``/api/v4`` internally.

    For unknown forge types the input *url* is returned unchanged.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    scheme = parsed.scheme or "https"
    if forge_type == "github":
        if host == "github.com" or host == "api.github.com":
            return "https://api.github.com"
        # GitHub Enterprise.  ``host`` already has any leading ``api.``
        # stripped during parsing of the user-facing URL.
        return f"{scheme}://{host}/api/v3"
    if forge_type == "gitlab":
        # python-gitlab takes the instance URL and adds ``/api/v4``.
        # Strip any ``/api/v4`` suffix the operator may have included
        # by accident (this used to be required in the old config
        # shape, so dogfood configs in the wild may still carry it).
        path = parsed.path or ""
        if path.rstrip("/").endswith("/api/v4"):
            path = path[: path.rfind("/api/v4")]
        if path:
            return f"{scheme}://{host}{path}"
        return f"{scheme}://{host}"
    return url


# ---------------------------------------------------------------------------
# Fork URL parsing
# ---------------------------------------------------------------------------

class ForkLocation(BaseModel):
    """Result of parsing a fork's human-facing URL.

    Carries the forge-native project handle (``owner/repo`` for GitHub,
    full path ``group/subgroup/project`` for GitLab) plus the canonical
    HTTPS clone URL.
    """

    native_id: str
    clone_url: str


def parse_fork_url(forge_type: str, url: str) -> ForkLocation:
    """Parse a human-facing fork URL into ``(native_id, clone_url)``.

    - **GitHub**: ``https://github.com/owner/repo`` ->
      ``native_id="owner/repo"``, ``clone_url="https://github.com/owner/repo.git"``.
    - **GitLab**: ``https://gitlab.com/group/subgroup/project`` ->
      ``native_id="group/subgroup/project"``,
      ``clone_url="https://gitlab.com/group/subgroup/project.git"``.

    Trailing ``.git`` and any ``/-/...`` GitLab path suffixes (e.g.
    ``/issues/1``) are stripped before parsing.

    Raises :class:`ValueError` when the URL has no project path.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    scheme = parsed.scheme or "https"
    raw_path = (parsed.path or "").strip("/")
    if not raw_path:
        raise ValueError(
            f"Cannot parse fork URL {url!r}: no project path."
        )

    # GitLab project paths can be followed by ``/-/issues/...`` etc.;
    # truncate at the ``/-/`` boundary so we keep just the project
    # path.  (GitHub uses query-style refs like ``/issues/1`` directly
    # under the project path, so we don't try to strip those here.)
    dash_marker = "/-/"
    dash_at = raw_path.find(dash_marker)
    if dash_at >= 0:
        raw_path = raw_path[:dash_at]

    if raw_path.endswith(".git"):
        raw_path = raw_path[: -len(".git")]

    if forge_type == "github":
        # GitHub project paths are always two segments.
        segments = raw_path.split("/")
        if len(segments) < 2:
            raise ValueError(
                f"GitHub project URL {url!r} must have an "
                "'owner/repo' path."
            )
        native_id = "/".join(segments[:2])
    elif forge_type == "gitlab":
        # GitLab paths can be arbitrarily deep due to subgroups.
        if "/" not in raw_path:
            raise ValueError(
                f"GitLab project URL {url!r} must include a group "
                "and project (e.g. 'group/project')."
            )
        native_id = raw_path
    else:
        # Unknown forge: pass the path through verbatim.  This still
        # produces a usable clone URL, and forge-specific tools can
        # decide what to do with the native_id.
        native_id = raw_path

    clone_url = f"{scheme}://{host}/{native_id}.git"
    return ForkLocation(native_id=native_id, clone_url=clone_url)


# ---------------------------------------------------------------------------
# $ENV_VAR expansion (deprecated; retained for any out-of-tree caller)
# ---------------------------------------------------------------------------
#
# The framework no longer expands ``$ENV_VAR`` strings in config
# files: credentials carry an explicit ``env_var_name`` field.  This
# helper is left in place because removing it eagerly would break
# any custom code that imported it; new code should not call it.


def expand_env_vars(data: Any) -> Any:
    """Recursively expand ``$ENV_VAR`` references in string values.

    A string value whose entire content matches ``$NAME`` is replaced
    with ``os.environ[NAME]``.  Non-string values and strings that do
    not start with ``$`` are returned unchanged.

    Dicts and lists are traversed recursively; all other types pass
    through unmodified.

    Raises :class:`ValueError` when a referenced variable is not set.
    """
    if isinstance(data, str):
        if data.startswith("$"):
            var_name = data[1:]
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(
                    f"Environment variable {var_name!r} "
                    f"(referenced as {data!r}) is not set"
                )
            return value
        return data
    if isinstance(data, dict):
        return {k: expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [expand_env_vars(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class ForgeSpec(BaseModel):
    """One entry in the ``"forges"`` array of ``gateway.json``.

    Most fields are optional; when omitted they are derived from
    ``url``:

    - ``name`` defaults to the URL hostname (e.g. ``github.com`` ->
      ``"github"``, ``gitlab.example.com`` ->
      ``"gitlab-example-com"``).
    - ``type`` is required for non-canonical hosts but may be omitted
      for ``github.com`` and ``gitlab.com``.
    - ``api_url`` is derived from ``url`` and ``type``; only set this
      to override (e.g. for non-standard GHE deployments).

    The ``url`` field is the human-facing instance URL (e.g.
    ``"https://github.com"``).  The legacy ``base_url`` field (which
    held the API URL) has been removed.
    """

    url: str = Field(
        default="",
        description="Human-facing instance URL of the forge",
    )
    type: str = Field(
        default="",
        description=(
            "Forge backend: 'github' or 'gitlab'.  Optional for "
            "well-known hosts (github.com, gitlab.com)."
        ),
    )
    name: str = Field(
        default="",
        description=(
            "Logical name of the forge entry (referenced from agent "
            "accounts).  Derived from `url` when omitted."
        ),
    )
    api_url: str = Field(
        default="",
        description=(
            "API base URL override.  Derived from `url` and `type` "
            "when omitted."
        ),
    )
    poll_interval: int = Field(
        default=30,
        ge=5,
        description="Seconds between event polling cycles",
    )
    unknown_actor_policy: UnknownActorPolicyMode = Field(
        default=UnknownActorPolicyMode.READ_ONLY,
        description=(
            "How incoming events from actors that are not declared "
            "as peers are handled on this forge.  'read_only' "
            "(the default) delivers structural events with an "
            "untrusted-content banner and drops conversational "
            "events.  'drop' drops every unknown-actor event.  "
            "'allow_response' delivers unknown-actor events with "
            "response-only constraints."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_unknown_actor_policy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy_key = "deliver_structural_from_non_peers"
        if legacy_key not in data:
            return data
        if "unknown_actor_policy" in data:
            raise ValueError(
                "ForgeSpec cannot specify both "
                "`unknown_actor_policy` and legacy "
                "`deliver_structural_from_non_peers`."
            )
        migrated = dict(data)
        legacy_value = migrated.pop(legacy_key)
        if legacy_value is True:
            migrated["unknown_actor_policy"] = UnknownActorPolicyMode.READ_ONLY.value
            return migrated
        if legacy_value is False:
            migrated["unknown_actor_policy"] = UnknownActorPolicyMode.DROP.value
            return migrated
        raise ValueError(
            "`deliver_structural_from_non_peers` must be a boolean when "
            "used as a legacy alias for `unknown_actor_policy`."
        )

    @model_validator(mode="after")
    def _fill_defaults(self) -> ForgeSpec:
        """Fill in derivable defaults so downstream code can rely on them.

        - When ``type`` is empty, infer it from the URL's host (for
          known hosts only; otherwise we leave it empty and the
          load-time validator raises a helpful error).
        - When ``name`` is empty, derive it from the URL.
        - When ``api_url`` is empty, derive it from ``url`` + ``type``.
        """
        if not self.url and not self.api_url:
            raise ValueError(
                "ForgeSpec requires at least one of `url` or `api_url`."
            )

        # Infer the type from URL when omitted (well-known hosts only).
        if not self.type and self.url:
            inferred = derive_forge_type_from_url(self.url)
            if inferred is None:
                raise ValueError(
                    f"Cannot infer forge type from URL {self.url!r}: "
                    "host is not a well-known forge host (github.com, "
                    "gitlab.com).  Set the `type` field explicitly."
                )
            self.type = inferred

        # Derive name from URL.
        if not self.name and self.url:
            self.name = derive_forge_name_from_url(self.url)

        # Derive API URL from instance URL + type.
        if not self.api_url and self.url and self.type:
            self.api_url = derive_api_url(self.type, self.url)

        return self


class ForkSpec(BaseModel):
    """A single fork of a project, hosted on a forge.

    The only required field is ``url`` -- the human-facing project URL
    on the forge (e.g. ``"https://github.com/owner/repo"``).  Both the
    forge and the native project handle are derived from it.

    Optional fields:

    - ``name``: the local git remote name.  Defaults to the forge
      name (single-fork case can fall back to ``"origin"`` -- see
      :meth:`ProjectSpec.resolved_forks`).
    - ``forge``: the name of the :class:`ForgeSpec` entry that hosts
      this fork.  Inferred from the URL host when omitted.
    - ``native_id``: optional forge-native project identifier override.
      Normally this is parsed from ``url``.  For GitLab, Thorn first
      uses the human-readable project path and can resolve it to a
      numeric project ID at runtime if a self-hosted instance rejects
      path-based API project lookups.  Set ``native_id`` only when that
      resolution is unavailable and the operator already knows the
      numeric project ID.
    - ``default_branch``: per-fork override for the default branch.
      When omitted, falls back to :attr:`ProjectSpec.default_branch`,
      and ultimately to a live lookup against the forge.
    """

    url: str = Field(
        description="Human-facing URL of the fork on its forge.",
    )
    name: str = Field(
        default="",
        description=(
            "Local git remote name.  Defaults to the forge's name "
            "(or 'origin' for the single-fork case)."
        ),
    )
    forge: str = Field(
        default="",
        description=(
            "Name of the ForgeSpec hosting this fork.  Inferred from "
            "the URL host when omitted."
        ),
    )
    native_id: str = Field(
        default="",
        description=(
            "Forge-native project identifier override.  When empty, "
            "derived from `url`; for GitLab, path-derived IDs may be "
            "resolved to numeric IDs at API-call time."
        ),
    )
    default_branch: str = Field(
        default="",
        description=(
            "Per-fork override for the project's default branch.  "
            "When omitted, falls back to the project-level default "
            "and ultimately to a live forge API lookup."
        ),
    )


class ProjectSpec(BaseModel):
    """One entry in the ``"projects"`` array of ``gateway.json``.

    A project has one or more forks.  The simplest single-fork case is
    just ``{ "name": "...", "url": "..." }``: a single fork is
    synthesized from ``url``.  Multi-fork projects use the ``forks``
    array.

    ``default_branch`` is optional.  When omitted, the gateway will
    look it up from the primary fork's forge on first access (cached
    for the process lifetime).
    """

    name: str
    url: str = Field(
        default="",
        description=(
            "Shorthand for a single-fork project.  Equivalent to "
            "writing a single-element `forks: [{ url: ... }]`."
        ),
    )
    native_id: str = Field(
        default="",
        description=(
            "Shorthand for a single-fork native_id override.  Only "
            "valid with top-level `url`; multi-fork projects should "
            "put native_id on the relevant fork."
        ),
    )
    default_branch: str = Field(
        default="",
        description=(
            "Project-level default branch override.  When omitted, "
            "looked up from the primary fork's forge on first access."
        ),
    )
    forks: list[ForkSpec] = Field(
        default_factory=list,
        description="Explicit list of forks (multi-fork projects).",
    )

    @model_validator(mode="after")
    def _check_url_xor_forks(self) -> ProjectSpec:
        if self.url and self.forks:
            raise ValueError(
                f"Project {self.name!r} cannot specify both a top-level "
                "`url` and a `forks` array.  Use `url` for the "
                "single-fork shorthand or `forks` for the explicit form."
            )
        if self.native_id and self.forks:
            raise ValueError(
                f"Project {self.name!r} cannot specify top-level "
                "`native_id` with a `forks` array.  Put native_id on "
                "the specific fork instead."
            )
        if not self.url and not self.forks:
            raise ValueError(
                f"Project {self.name!r} must specify either `url` (for "
                "a single fork) or a non-empty `forks` array."
            )
        return self

    def resolved_forks(self) -> list[ForkSpec]:
        """Return the effective fork list (synthesizing from `url` when needed)."""
        if self.forks:
            return list(self.forks)
        return [ForkSpec(url=self.url, native_id=self.native_id)]


class PlannedEgressAllowlistEntry(BaseModel):
    """A future direct-egress exception for sandbox containers.

    Thorn does not enforce per-host direct egress today.  The active
    boundary is :attr:`SandboxConfig.egress_network`: when the bundled
    broker provides an internal OCI network, containers can reach the
    broker and nothing else directly.  This entry records operator
    intent for a future allow-list implementation without presenting
    that intent as an active security control.

    Both fields are explicit:

    * ``host`` -- DNS name or IP literal.  No wildcard / pattern
      support: each upstream that might need direct access gets its
      own entry, so a future audit of the allow-list is a literal
      enumeration rather than a regex review.
    * ``port`` -- TCP port.  Always required; we deliberately do
      not default to 443 because "a planned exception to the
      broker-only policy" warrants explicit attention to the port.
    """

    host: str = Field(
        min_length=1,
        description="DNS hostname or IP literal of the upstream.",
    )
    port: int = Field(
        ge=1, le=65535,
        description="TCP port the sandbox may connect to on this host.",
    )


class SandboxConfig(BaseModel):
    """Agency-wide defaults for the per-agent sandbox container.

    Sandboxing has two configuration layers: an agency-wide default (this
    model, attached to :class:`GatewayConfig`) and an optional per-agent
    override (the ``sandbox`` block of ``agent.json``). A field set on the
    agent wins; otherwise the agency default applies.

    Fields are intentionally named after operator concepts rather
    than implementation details so that diagnosing why a container
    behaved a certain way amounts to reading this block.

    The model defaults to the container backend. :class:`GatewayConfig`
    materializes this model even when a parsed agency configuration omits its
    ``sandbox`` block, making container execution the gateway's secure default.
    Operators can explicitly select ``backend: "subprocess"``. Local CLI and
    direct :class:`~thorn.runtime.Runtime` construction pass no agency-wide
    sandbox model and use the subprocess fallback in
    :mod:`thorn.sandbox._resolve` instead.
    """

    backend: Literal["subprocess", "container"] = Field(
        default="container",
        description=(
            "Sandbox executor backend.  'subprocess' keeps the Phase-A "
            "behavior (toolhost runs as a host subprocess).  'container' "
            "runs the toolhost inside an OCI container per agent."
        ),
    )
    oci_runtime: Literal["podman", "docker"] | None = Field(
        default=None,
        description=(
            "OCI runtime to use ('podman' or 'docker').  When null, "
            "auto-detect: prefer podman, fall back to docker.  Has "
            "no effect when backend is 'subprocess'."
        ),
    )
    image: str = Field(
        default="",
        description=(
            "Default container image tag (e.g. 'thorn-sandbox:0.1.0').  "
            "When empty, the framework default 'thorn-sandbox:<thorn-version>' "
            "applies.  Per-agent agent.json sandbox.image overrides this."
        ),
    )
    env_passthrough: list[str] = Field(
        default_factory=list,
        description=(
            "Names of host environment variables forwarded into every "
            "agent container.  Operator-controlled allow-list; per-agent "
            "agent.json may *add* to this list, never *remove* from it."
        ),
    )
    dev_mount_runtime: bool = Field(
        default=False,
        description=(
            "If true, bind-mount this checkout's thorn source tree "
            "read-only into each container at /opt/thorn-runtime and set "
            "PYTHONPATH so the in-container daemon picks up local edits "
            "without rebuilding the image.  Off by default; opt-in for "
            "developer workflows."
        ),
    )
    container_ready_timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description=(
            "Stage-one readiness budget: seconds to wait for the "
            "container to reach 'running' after launch.  Distinct from "
            "the daemon's socket-bind poll, which is governed by the "
            "executor."
        ),
    )
    egress_network: str | None = Field(
        default=None,
        description=(
            "Phase D: name of the OCI network the sandbox container "
            "joins.  When set, ``--network <name>`` is passed at "
            "container launch and the agent's outbound is bounded by "
            "what that network can reach.  Operators creating the "
            "network with ``--internal`` (so it has no NAT to the "
            "host) and connecting only the broker to it implements "
            "the 'broker-only' egress policy; the bundled "
            "docker-compose deploys exactly this shape under the "
            "name 'thorn-broker'.  Default null keeps the Phase-B "
            "behavior (default OCI bridge, full host network access)."
        ),
    )
    planned_egress_allowlist: list[PlannedEgressAllowlistEntry] = Field(
        default_factory=list,
        description=(
            "Future direct-egress exceptions as ``(host, port)`` "
            "pairs.  This field has no runtime effect today: sandbox "
            "outbound traffic is controlled only by "
            ":attr:`egress_network` and the network topology behind "
            "it.  When the gateway sees a non-empty planned list it "
            "logs a startup warning so the inactive security posture "
            "is visible to operators."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def reject_active_egress_allowlist_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and "egress_allowlist" in data:
            raise ValueError(
                "sandbox.egress_allowlist was removed because Thorn "
                "does not enforce per-host direct egress yet. Use "
                "sandbox.planned_egress_allowlist to record future "
                "exceptions; it has no runtime effect today.",
            )
        return data

    # ------------------------------------------------------------------
    # Phase E hardening fields
    #
    # Conservative defaults match the current threat model
    # (docs/threat-model.md): request cap-drop=ALL, enable
    # no-new-privileges, run with a read-only rootfs (with tmpfs scratch
    # space), and apply 2 GiB / 2 CPU / 512 pid limits. The container
    # host adds only the short-lived entrypoint capabilities needed to
    # install the broker CA and transition to the gateway operator's identity.
    # Operators with heavier workloads expand these per agent in
    # ``agent.json sandbox`` (see :class:`AgentSandboxOverride`).
    # ------------------------------------------------------------------

    capabilities_drop: list[str] = Field(
        default_factory=lambda: ["ALL"],
        description=(
            "Linux capability names dropped from every "
            "sandbox container's bounding set (each entry becomes "
            "``--cap-drop=<name>``).  The literal ``\"ALL\"`` drops "
            "every capability the runtime would otherwise grant. The "
            "container host separately adds the minimal capabilities "
            "needed by the root entrypoint, which clears its bounding "
            "set before executing the operator-identity toolhost.  "
            "Per-agent ``agent.json sandbox.capabilities_drop`` "
            "extends this list additively (see "
            ":class:`AgentSandboxOverride`)."
        ),
    )
    capabilities_add: list[str] = Field(
        default_factory=list,
        description=(
            "Phase E: capability names granted *after* "
            ":attr:`capabilities_drop`.  Used for agents that "
            "legitimately need a specific cap (e.g. "
            "``\"NET_RAW\"`` for ``ping``).  Adding caps is a "
            "deliberate departure from the conservative default "
            "and should be justified per agent."
        ),
    )
    security_opts: list[str] = Field(
        default_factory=lambda: ["no-new-privileges"],
        description=(
            "Phase E: values passed through ``--security-opt=<value>``. "
            "The default ``no-new-privileges`` prevents setuid binaries "
            "from elevating, even if one ends up in a derived image.  "
            "Operators with their own profiles (AppArmor, custom "
            "seccomp) extend this list per agency or per agent."
        ),
    )
    read_only_root: bool = Field(
        default=True,
        description=(
            "Phase E: when true, mount the container's root "
            "filesystem read-only (``--read-only``) and provide "
            "scratch tmpfs at ``/tmp`` and ``/var/tmp`` so tools "
            "writing to those paths continue to work.  The default "
            "is on because the agent's writable footprint is "
            "intentionally limited to the bind-mounted "
            "``/agent/{home,workspace,control}`` paths.  Per-agent "
            "override exists for tool ecosystems that legitimately "
            "need a writable rootfs (rare; typically dogfooding or "
            "specialised toolchains)."
        ),
    )
    memory_limit: str | None = Field(
        default="2G",
        description=(
            "Phase E: maximum memory the container may use, in the "
            "form accepted by ``podman``/``docker --memory`` (e.g. "
            "``\"2G\"``, ``\"512M\"``).  Defaults to ``\"2G\"`` so a "
            "leaking tool OOMs the container before crowding the "
            "host.  Per-agent override raises (or lowers) this; set "
            "to ``null`` to remove the cap."
        ),
    )
    cpu_limit: float | None = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Phase E: maximum fractional CPU the container may "
            "consume (``--cpus``).  Defaults to ``2.0``.  Per-agent "
            "override raises (or lowers) this; set to ``null`` to "
            "remove the cap."
        ),
    )
    pid_limit: int | None = Field(
        default=512,
        ge=0,
        description=(
            "Phase E: maximum number of processes the container's "
            "pid namespace may hold (``--pids-limit``).  Defaults "
            "to ``512`` -- roomy for shell + git + Python + a "
            "couple of MCP servers, tight enough that a fork bomb "
            "trips the limit before nuking the host.  Set to "
            "``null`` to remove the cap."
        ),
    )


class AgentSandboxOverride(BaseModel):
    """Per-agent overrides for the agency's :class:`SandboxConfig`.

    Lives in the ``sandbox`` block of an agent's ``agent.json``.
    Every field is optional; an absent field falls back to the agency
    default (which itself may fall back to the framework default).
    Setting a field to its explicit empty form (``[]`` for the list,
    ``""`` for ``image``) does **not** reset to the agency value --
    it is a literal override.

    The merge rules differ slightly across fields:

    * ``image`` -- replaces the agency value when set (most common
      reason to override per-agent: a specialized agent that needs
      extra system packages baked into its sandbox image).
    * ``env_passthrough`` -- *added* to the agency list, never
      replaces it.  Operators can broaden the allow-list per agent
      but never narrow it without a config change at the agency
      level.  This avoids the surprise of "I added LANG to the
      agency-wide list and a particular agent silently lost it".
    * Other fields -- replace the agency value verbatim when set.

    The merge happens in :func:`thorn.runtime._sandbox.resolve_sandbox_config`,
    not here; this class is just the on-disk shape.
    """

    backend: Literal["subprocess", "container"] | None = Field(
        default=None,
        description=(
            "Per-agent override for the executor backend.  When null, "
            "use the agency default."
        ),
    )
    image: str | None = Field(
        default=None,
        description=(
            "Per-agent override for the container image.  When null, "
            "use the agency default; when a non-empty string, replaces "
            "the agency value."
        ),
    )
    env_passthrough: list[str] = Field(
        default_factory=list,
        description=(
            "Additional env-var names this agent's container should "
            "see (added to the agency-wide allow-list, not replacing it)."
        ),
    )
    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Literal env entries added to this agent's container, "
            "not read from the host process environment."
        ),
    )
    container_ready_timeout_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Per-agent override for the stage-one readiness timeout."
        ),
    )

    # ------------------------------------------------------------------
    # Phase E hardening overrides
    #
    # The merge rule for each field is documented on
    # :func:`thorn.sandbox._resolve.resolve_sandbox_config`.  Lists
    # are *additive* (agency + agent, dedup, agency-first); scalars
    # replace when the agent value is non-``None``.
    #
    # The per-agent override surface is broad because both the agency
    # configuration and ``agent.json`` are operator-controlled. These
    # overrides are a convenience knob, not a security boundary. Agencies
    # that want a uniform policy simply do not set per-agent fields.
    # ------------------------------------------------------------------

    capabilities_drop: list[str] = Field(
        default_factory=list,
        description=(
            "Capability names this agent additionally drops, on top "
            "of the agency's ``capabilities_drop`` list.  Additive: "
            "an empty list means \"no addition\" rather than \"reset "
            "agency to nothing\".  Including ``\"ALL\"`` here when "
            "the agency does not is the way to opt a single agent "
            "into the conservative default."
        ),
    )
    capabilities_add: list[str] = Field(
        default_factory=list,
        description=(
            "Capability names this agent adds on top of the agency's "
            "``capabilities_add`` list.  Additive."
        ),
    )
    security_opts: list[str] = Field(
        default_factory=list,
        description=(
            "``--security-opt`` values this agent adds on top of the "
            "agency's list.  Additive."
        ),
    )
    read_only_root: bool | None = Field(
        default=None,
        description=(
            "Per-agent override for the read-only-rootfs policy.  "
            "``None`` (the default) keeps the agency value; "
            "``False`` opts a single agent out of the agency's "
            "default-on policy (useful for dogfooding agents that "
            "need a writable rootfs); ``True`` opts a single agent "
            "in when the agency has it disabled."
        ),
    )
    memory_limit: str | None = Field(
        default=None,
        description=(
            "Per-agent override for ``--memory``.  ``None`` keeps "
            "the agency value; a string replaces it; explicitly "
            "passing ``null`` in the JSON does **not** remove the "
            "limit -- the agency value is still inherited.  To "
            "remove the cap for a single agent, set the agency "
            "field to ``null`` and let the agent inherit."
        ),
    )
    cpu_limit: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Per-agent override for ``--cpus``.  ``None`` keeps the "
            "agency value; a float replaces it."
        ),
    )
    pid_limit: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Per-agent override for ``--pids-limit``.  ``None`` "
            "keeps the agency value; an integer replaces it."
        ),
    )


BrokerMode = Literal["bundled", "external"]
"""Discriminator for :class:`BrokerConfig.mode`.

* ``"bundled"`` (default): ``thorn serve`` brings up its own OneCLI
  + Postgres compose stack on startup, mints an admin key, and tears
  the stack down on shutdown.  Operators do not have to write
  ``admin_url`` / ``admin_api_key`` / ``proxy_url`` themselves; the
  bundled supervisor synthesises those at runtime.
* ``"external"``: the operator runs OneCLI themselves (bare metal,
  Kubernetes, an existing compose deployment, etc.) and points
  Thorn at it via explicit ``admin_url`` / ``admin_api_key`` /
  ``proxy_url``.  This is the pre-bundled-broker integration shape
  preserved for advanced setups.
"""


THORN_BUNDLED_BROKER_ONECLI_IMAGE_ENV_VAR = "THORN_BUNDLED_BROKER_ONECLI_IMAGE"
"""Environment variable used to override the bundled OneCLI image."""

THORN_BUNDLED_BROKER_POSTGRES_IMAGE_ENV_VAR = (
    "THORN_BUNDLED_BROKER_POSTGRES_IMAGE"
)
"""Environment variable used to override the bundled Postgres image."""

THORN_BUNDLED_BROKER_PROXY_IMAGE_ENV_VAR = "THORN_BUNDLED_BROKER_PROXY_IMAGE"
"""Environment variable used to override the bundled sandbox proxy image."""


def _is_loopback_host(host: str) -> bool:
    normalized = host.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _admin_url_uses_allowed_transport(admin_url: str) -> bool:
    parsed = urlparse(admin_url)
    if parsed.scheme.lower() != "http":
        return True
    hostname = parsed.hostname
    if hostname is None:
        return False
    return _is_loopback_host(hostname)


class OCIImageReference(str):
    """A concrete OCI image reference used by bundled broker config.

    Thorn does not parse image references into registry/name/tag/digest
    components because compose accepts the full OCI reference string
    directly and registries have a broad grammar.  The explicit type
    still gives image references a named boundary in the gateway schema
    and rejects values that cannot be valid compose image references.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str.__str__,
                return_schema=core_schema.str_schema(),
                when_used="always",
            ),
        )

    @classmethod
    def _validate(cls, value: Any) -> "OCIImageReference":
        if isinstance(value, OCIImageReference):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"OCIImageReference requires str input, "
                f"got {type(value).__name__}"
            )
        if not value:
            raise ValueError("OCI image reference must not be empty")
        if value.strip() != value or any(char.isspace() for char in value):
            raise ValueError(
                "OCI image reference must not contain whitespace",
            )
        return cls(value)


class BundledBrokerImageConfig(BaseModel):
    """Optional image references for Thorn's bundled broker compose stack.

    Each field is an override.  Omitted fields fall back to the host
    environment variables consumed by the bundled compose file, and
    then to the defaults baked into that compose resource.
    """

    onecli: OCIImageReference | None = Field(
        default=None,
        description=(
            "Optional OCI image reference for the bundled OneCLI "
            "service.  When omitted, the bundled compose file uses "
            f"${THORN_BUNDLED_BROKER_ONECLI_IMAGE_ENV_VAR} if set, "
            "otherwise its built-in default."
        ),
    )
    postgres: OCIImageReference | None = Field(
        default=None,
        description=(
            "Optional OCI image reference for the bundled Postgres "
            "service.  When omitted, the bundled compose file uses "
            f"${THORN_BUNDLED_BROKER_POSTGRES_IMAGE_ENV_VAR} if set, "
            "otherwise its built-in default."
        ),
    )
    proxy: OCIImageReference | None = Field(
        default=None,
        description=(
            "Optional OCI image reference for the bundled sandbox-facing "
            "TCP proxy service.  When omitted, the bundled compose file "
            f"uses ${THORN_BUNDLED_BROKER_PROXY_IMAGE_ENV_VAR} if set, "
            "otherwise its built-in default."
        ),
    )

    def has_overrides(self) -> bool:
        """Return true when at least one image reference is configured."""
        return (
            self.onecli is not None
            or self.postgres is not None
            or self.proxy is not None
        )

    def compose_env_overrides(self) -> dict[str, str]:
        """Return compose env entries represented by this config."""
        env: dict[str, str] = {}
        if self.onecli is not None:
            env[THORN_BUNDLED_BROKER_ONECLI_IMAGE_ENV_VAR] = str(self.onecli)
        if self.postgres is not None:
            env[THORN_BUNDLED_BROKER_POSTGRES_IMAGE_ENV_VAR] = str(self.postgres)
        if self.proxy is not None:
            env[THORN_BUNDLED_BROKER_PROXY_IMAGE_ENV_VAR] = str(self.proxy)
        return env


class BrokerConfig(BaseModel):
    """Agency-wide configuration for the OneCLI credential broker.

    The broker is the substitution proxy that per-agent sandbox
    containers funnel their outbound HTTPS through.  When this block
    is set with ``enabled: true`` (the default when ``GatewayConfig``
    auto-fills it), the gateway registers each agent's credentials
    with the broker at agent-load and the container env carries only
    placeholder values.  When ``enabled: false`` (or when the
    sandbox backend resolves to ``subprocess``), the broker is not
    used and credentials flow via the Phase-B env-injection path;
    the audit invariant is not enforced in that case.

    Two modes:

    * ``mode: "bundled"`` (default): ``thorn serve`` manages its own
      OneCLI stack via ``BundledBrokerSupervisor``.  ``admin_url``,
      ``admin_api_key``, and ``proxy_url`` MUST be left unset --
      they are filled in at runtime from the supervisor-discovered
      values.  ``bundled_images`` may optionally pin the OneCLI and
      Postgres image references used by that managed stack.
    * ``mode: "external"``: the operator points the gateway at an
      OneCLI deployment they manage themselves.  ``admin_url``,
      ``admin_api_key``, and ``proxy_url`` are all required.

    See [docs/aspirational/architecture.md] for the integration
    shape.  R1/R2 confirm: the proxy accepts ``Basic`` auth via
    embedded ``HTTPS_PROXY`` URL credentials (no per-tool bearer
    wiring needed); the admin surface is OneCLI's Next.js
    ``/api/agents``, ``/api/secrets``, and friends authenticated
    with ``Authorization: Bearer oc_<hex>``.
    """

    mode: BrokerMode = Field(
        default="bundled",
        description=(
            "How ``thorn serve`` obtains the broker.  ``\"bundled\"`` "
            "(the default) brings up a per-process OneCLI compose "
            "stack on startup and tears it down on shutdown.  "
            "``\"external\"`` connects to an OneCLI deployment the "
            "operator is responsible for; in that case ``admin_url``, "
            "``admin_api_key``, and ``proxy_url`` are all required."
        ),
    )
    enabled: bool = Field(
        default=True,
        description=(
            "When false, the broker is treated as not configured even "
            "if other fields are set; agent-load uses Phase-B env "
            "injection and the audit invariant is not enforced.  "
            "Useful for temporarily disabling the broker without "
            "removing the configuration -- typically paired with "
            "sandbox.backend = 'subprocess', in which case the "
            "agency runs without isolation or broker."
        ),
    )
    admin_url: str = Field(
        default="",
        description=(
            "Base URL of OneCLI's admin/management API (the Next.js "
            "service, default port 10254).  The gateway calls this "
            "to create agent identities, register secrets, and mint "
            "per-agent proxy tokens.  Required when ``mode == "
            "\"external\"``; MUST be left empty when ``mode == "
            "\"bundled\"`` (the supervisor fills it in).  Example: "
            "'https://onecli-web:10254'."
        ),
    )
    admin_api_key_env_var: str | None = Field(
        default=None,
        description=(
            "Name of the environment variable holding the OneCLI "
            "admin API key (Bearer token, prefix 'oc_'), used by "
            "the gateway as ``Authorization`` when driving the "
            "admin API.  The literal value is read from "
            "``os.environ`` at gateway startup; the gateway state "
            "never persists the literal.  Required when ``mode == "
            "\"external\"``; MUST be left unset when ``mode == "
            "\"bundled\"`` (the supervisor mints the key in process "
            "memory)."
        ),
    )
    proxy_url: str = Field(
        default="",
        description=(
            "URL of OneCLI's HTTP/HTTPS substitution proxy (the Rust "
            "gateway, default port 10255).  Each per-agent sandbox "
            "container's ``HTTPS_PROXY`` env var points at this URL "
            "with the per-agent proxy token embedded as Basic-auth "
            "credentials in the URL.  Required when ``mode == "
            "\"external\"``; MUST be left empty when ``mode == "
            "\"bundled\"`` (the supervisor fills it in).  Example: "
            "'http://onecli-gateway:10255'."
        ),
    )
    ca_certificate_path: str | None = Field(
        default=None,
        description=(
            "Optional host filesystem path where the gateway should "
            "write OneCLI's MITM CA certificate (PEM).  When ``None`` "
            "(the default), the gateway derives a path under the "
            "agency home (``<agency_home>/onecli-ca.pem``).  At "
            "startup, the gateway always pulls the CA from "
            "``GET /api/gateway/ca`` and writes it to this path; the "
            "file is then bind-mounted read-only into every per-agent "
            "sandbox container so its TLS stacks trust the broker's "
            "MITM certificates.  Setting this field is only necessary "
            "when an operator wants the CA at a specific path (e.g. "
            "to share it with non-Thorn tooling); the default keeps "
            "the deployment story simple -- the gateway owns CA "
            "acquisition and the operator does not need to wire any "
            "shared volumes."
        ),
    )
    bundled_images: BundledBrokerImageConfig = Field(
        default_factory=BundledBrokerImageConfig,
        description=(
            "Optional OCI image references for the bundled OneCLI + "
            "Postgres compose stack.  Applies only when mode is "
            "'bundled'; external brokers are operator-managed and "
            "must not set this."
        ),
    )

    @model_validator(mode="after")
    def _check_mode_invariants(self) -> BrokerConfig:
        """Enforce the bundled/external invariants documented above.

        Bundled mode is "the supervisor fills everything in", so
        carrying any of the URL / key fields is a configuration
        mistake (the operator typed values that the runtime is going
        to overwrite, which is confusing and indicates they probably
        wanted ``mode: external``).  External mode requires all three
        because the supervisor will not run, so missing values would
        only surface as a confusing 404 / connection-refused later.
        """
        if self.mode == "bundled":
            stray: list[str] = []
            if self.admin_url:
                stray.append("admin_url")
            if self.admin_api_key_env_var is not None:
                stray.append("admin_api_key_env_var")
            if self.proxy_url:
                stray.append("proxy_url")
            if stray:
                raise ValueError(
                    f"broker.mode='bundled' must not carry "
                    f"{', '.join(stray)}; those fields are filled in "
                    "at startup by the bundled-broker supervisor.  "
                    "Drop the field(s), or switch to "
                    "broker.mode='external' if you intend to point "
                    "Thorn at an OneCLI deployment you manage yourself."
                )
        else:  # external
            missing: list[str] = []
            if not self.admin_url:
                missing.append("admin_url")
            if self.admin_api_key_env_var is None:
                missing.append("admin_api_key_env_var")
            if not self.proxy_url:
                missing.append("proxy_url")
            if missing:
                raise ValueError(
                    f"broker.mode='external' requires {', '.join(missing)}; "
                    "set them in gateway.json.  Switch to "
                    "broker.mode='bundled' (or omit the broker block "
                    "entirely) if you want `thorn serve` to manage the "
                    "broker for you."
                )
            if self.bundled_images.has_overrides():
                raise ValueError(
                    "broker.mode='external' must not carry bundled_images; "
                    "image selection belongs to the operator-managed "
                    "external broker deployment, not Thorn's bundled "
                    "compose stack."
                )
            if not _admin_url_uses_allowed_transport(self.admin_url):
                raise ValueError(
                    "broker.admin_url uses non-loopback HTTP.  Expose the "
                    "OneCLI admin API over HTTPS, bind it to a loopback "
                    "HTTP URL such as http://127.0.0.1:10254, or switch "
                    "to broker.mode='bundled' so Thorn can manage the "
                    "loopback-only admin endpoint."
                )
        return self


class GatewayConfig(BaseModel):
    """Top-level model for an agency configuration file.

    Carries the agency's workspace directory along with its declared
    forges and projects.  Future plug-in service categories will
    appear as additional typed array fields (for example,
    ``messaging_services: list[...]``).

    Defaults are deliberately *secure*: an agency with no
    ``sandbox`` and no ``broker`` block boots with the Phase-E
    container sandbox + the bundled OneCLI broker.  Operators who
    need to opt out of containerised execution set
    ``sandbox.backend = "subprocess"`` explicitly; the model
    validator then drops the broker default (a broker without a
    container has nothing to inject the proxy into).

    Note: these schema-level defaults only apply when
    :func:`load_gateway_config` parses an actual agency config file,
    which today is done by ``thorn serve`` and related agency commands.  The
    in-process CLI entry points (``thorn run`` / ``thorn chat``)
    construct ``Runtime`` with ``sandbox_config=None``, which falls
    through to the unchanged
    :data:`thorn.sandbox._resolve._DEFAULT_AGENCY_SANDBOX` (still
    ``backend="subprocess"``).  Flipping schema defaults here does
    not silently change CLI behavior.
    """

    workspace: str = Field(
        default="",
        description=(
            "Filesystem path to the agency's workspace root.  Absolute "
            "paths are used as-is; relative paths are resolved against "
            "the directory containing gateway.json (the agency home).  "
            "Empty string means no workspace was configured at bootstrap; "
            "in that case 'thorn serve' must be given an explicit "
            "--workspace override."
        ),
    )
    forges: list[ForgeSpec] = Field(default_factory=list)
    projects: list[ProjectSpec] = Field(default_factory=list)
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description=(
            "Agency-wide default LLM provider and model settings.  "
            "Agent agent.json files may override this with their own "
            "`llm` block; secrets remain environment-variable references."
        ),
    )
    peers: list[PeerSpec] = Field(
        default_factory=list,
        description=(
            "Operator-declared list of peers -- humans and bots whose "
            "messages the gateway is willing to treat as instructions "
            "for an agent.  An empty list means strict trust: every "
            "conversational event from anyone other than the agent "
            "itself is dropped at the event boundary, and every "
            "structural event renders with a non-peer banner.  See "
            "the threat-model docs for details on what peerhood "
            "actually grants."
        ),
    )
    sandbox: SandboxConfig | None = Field(
        default=None,
        description=(
            "Agency-wide sandbox defaults.  When omitted from "
            "gateway.json, defaults to a fresh ``SandboxConfig()`` "
            "with the Phase-E container backend on; set explicitly "
            "to ``{\"backend\": \"subprocess\"}`` to opt out of "
            "containerised sandboxing (which also disables the "
            "bundled broker, since there is no container to "
            "inject the proxy into)."
        ),
    )
    broker: BrokerConfig | None = Field(
        default=None,
        description=(
            "Agency-wide credential-broker configuration.  When "
            "omitted from gateway.json AND the resolved sandbox "
            "backend is ``container``, defaults to a fresh "
            "``BrokerConfig(mode='bundled')`` so ``thorn serve`` "
            "brings up its own OneCLI stack on startup.  When the "
            "sandbox backend resolves to ``subprocess``, defaults "
            "to ``None`` (no broker -- there's no container to "
            "inject into).  Set explicitly with ``mode: 'external'`` "
            "to point Thorn at an OneCLI deployment you manage "
            "yourself, or with ``enabled: false`` to keep the "
            "container backend without a broker."
        ),
    )

    @model_validator(mode="after")
    def _fill_secure_defaults(self) -> GatewayConfig:
        """Apply the "absent block -> secure default" rules.

        Two cases the schema cannot express via plain ``default=``:

        1. Sandbox block omitted -> default to a container-backend
           ``SandboxConfig()``.  We can't use a default-factory at
           the field level because we want ``"subprocess"`` to remain
           the default for the runtime-level
           :data:`thorn.sandbox._resolve._DEFAULT_AGENCY_SANDBOX`,
           which is what the in-process CLI paths fall back to.
           Flipping the GatewayConfig field default keeps the schema
           secure-by-default exactly when a ``gateway.json`` is the
           thing being loaded.
        2. Broker block omitted -> default to bundled, *but only
           when* the resolved sandbox backend is ``container``.  A
           bundled broker with a subprocess sandbox would be
           pointless (no container to inject the proxy into); the
           rule encodes that.

        Cross-config peer validation deliberately does **not** live
        here, even though it would be tempting to add as a third
        bullet.  The "every ``peer.account.service`` references a
        real service" check has to wait until forge synthesis has
        run -- otherwise a peer that names a forge derived from a
        project fork URL (the common case for an operator who never
        wrote an explicit ``forges:`` block) would spuriously fail.
        That check lives in :func:`_resolve_forges_and_projects`,
        where the resolved service-name set is in scope.  Per-peer
        schema invariants (id grammar, account fields non-empty)
        live on :class:`PeerSpec` itself; duplicate-id and
        per-account-collision checks live on :class:`PeerRegistry`.
        """
        if self.sandbox is None:
            self.sandbox = SandboxConfig()
        if self.broker is None and self.sandbox.backend == "container":
            self.broker = BrokerConfig(mode="bundled")
        return self

    def resolve_workspace(self, agency_home: Path) -> Path | None:
        """Resolve the configured workspace path against *agency_home*.

        Absolute paths are returned unchanged (after :meth:`Path.resolve`);
        relative paths are resolved relative to *agency_home*.  Returns
        ``None`` when no workspace is configured (``workspace == ""``),
        in which case the caller is expected to require an explicit
        override.
        """
        if not self.workspace:
            return None
        candidate = Path(self.workspace)
        if not candidate.is_absolute():
            candidate = agency_home / candidate
        return candidate.resolve()


# ---------------------------------------------------------------------------
# Forge inference and fork resolution
# ---------------------------------------------------------------------------


class ResolvedFork(BaseModel):
    """A fully-resolved fork with all derived fields filled in."""

    forge_name: str
    forge_type: str
    name: str
    native_id: str
    clone_url: str
    default_branch: str = ""


class ResolvedProject(BaseModel):
    """A fully-resolved project: forks tied to forge entries."""

    name: str
    default_branch: str = ""
    forks: list[ResolvedFork]


def _index_forges_by_host(
    forge_specs: list[ForgeSpec],
) -> dict[str, ForgeSpec]:
    """Index forge specs by URL hostname for fast fork-host lookup."""
    by_host: dict[str, ForgeSpec] = {}
    for fs in forge_specs:
        if not fs.url:
            continue
        host = _host_of(fs.url)
        if host:
            by_host[host] = fs
    return by_host


def validate_peers_against_services(
    peers: list[PeerSpec],
    service_names: Iterable[str],
    *,
    service_types_by_name: dict[str, str] | None = None,
    account_id_policy: PeerAccountIDPolicy = (
        PeerAccountIDPolicy.ALLOW_HANDLE_ONLY
    ),
) -> None:
    """Cross-check that every peer account references a real service.

    *service_names* must be the **resolved** set of service names --
    the union of operator-declared ``forges[].name`` entries and
    forges synthesised by :func:`_resolve_forges_and_projects` for
    project fork URLs that lacked an explicit forge entry.  Passing
    only the operator-declared list (e.g. ``GatewayConfig.forges``)
    is a layering bug: a peer that names a synthesized forge would
    spuriously fail validation.

    Per-peer schema checks (id grammar) and registry-level checks
    (duplicate ids, accounts claimed by two peers) live elsewhere
    -- on :class:`PeerSpec` and :class:`PeerRegistry` respectively.
    This function is the cross-config check that depends on having
    the resolved service-name set in hand, and is therefore not
    expressible as a pydantic post-validator on
    :class:`GatewayConfig` (which runs at JSON-load time, before
    forge synthesis).

    Args:
        peers: The :class:`PeerSpec` list from the gateway config.
        service_names: Resolved set of service names the gateway
            will instantiate.  Currently this is just resolved
            forge names; future service categories that grow
            peer-reference semantics will join the same set.
        service_types_by_name: Optional mapping from service name to
            forge type.  When supplied with
            :attr:`PeerAccountIDPolicy.REJECT_RESOLVABLE_HANDLES`,
            GitHub/GitLab peer accounts whose ``account_id`` is still
            a textual handle are rejected.
        account_id_policy: Whether this validation pass is allowed
            to see handle-only peer accounts.  Normal gateway startup
            rejects them; ``thorn serve resolve-peers`` uses the
            allow mode while it is rewriting the config.

    Raises:
        ValueError: If any peer has an account whose ``service``
            does not appear in *service_names*.

    Logs at WARNING when a peer has zero accounts, since no
    incoming event will ever match such a peer.  This is not a
    fatal error -- operators may be mid-edit -- but it is worth
    surfacing.
    """
    known = set(service_names)
    for peer in peers:
        for account in peer.accounts:
            if account.service in known:
                continue
            raise ValueError(
                f"Peer {peer.id!r} has an account on service "
                f"{account.service!r}, but no service with that "
                "name exists in this gateway (after forge "
                "synthesis from project fork URLs).  Either add a "
                "matching forge entry to the `forges:` array, "
                "declare a project whose fork URL implies that "
                "forge, or correct the peer's account.service.  "
                f"Known services: {sorted(known)}"
            )
        if not peer.accounts:
            log.warning(
                "Peer %r has no accounts declared; no incoming "
                "events will match this peer until at least one "
                "account is added.", peer.id,
            )
    if account_id_policy is PeerAccountIDPolicy.ALLOW_HANDLE_ONLY:
        return

    problems = collect_handle_only_peer_account_problems(
        peers,
        service_types_by_name or {},
    )
    if not problems:
        return
    rendered = "; ".join(
        (
            f"peer {problem.peer_id!r} service {problem.service_name!r} "
            f"has handle-only account_id {problem.account_id!r}"
        )
        for problem in problems
    )
    raise ValueError(
        "Peer accounts on GitHub/GitLab must use immutable platform "
        "user IDs as account_id values.  Run `thorn serve resolve-peers` "
        f"to rewrite textual handles before startup.  Unresolved: {rendered}"
    )


def collect_handle_only_peer_account_problems(
    peers: list[PeerSpec],
    service_types_by_name: dict[str, str],
) -> list[PeerAccountHandleOnlyProblem]:
    """Return GitHub/GitLab peer accounts still keyed by mutable handles."""
    problems: list[PeerAccountHandleOnlyProblem] = []
    for peer in peers:
        for account in peer.accounts:
            forge_type = service_types_by_name.get(account.service, "")
            if forge_type not in _FORGE_TYPES_WITH_IMMUTABLE_USER_IDS:
                continue
            if account.account_id.isdigit():
                continue
            problems.append(
                PeerAccountHandleOnlyProblem(
                    peer_id=peer.id,
                    service_name=account.service,
                    account_id=account.account_id,
                    forge_type=forge_type,
                )
            )
    return problems


def _resolve_forges_and_projects(
    config: GatewayConfig,
    *,
    peer_account_id_policy: PeerAccountIDPolicy = (
        PeerAccountIDPolicy.REJECT_RESOLVABLE_HANDLES
    ),
) -> tuple[list[ForgeSpec], list[ResolvedProject]]:
    """Resolve forge entries and projects for *config*.

    - Walks each project's forks to ensure a matching :class:`ForgeSpec`
      exists; synthesizes one for any well-known host that lacks an
      explicit entry.
    - Replaces each :class:`ForkSpec` with a :class:`ResolvedFork` whose
      ``forge``, ``name``, ``native_id``, and ``clone_url`` are filled in.
    - Detects collisions where two different forge specs end up with
      the same ``name``.
    - Validates :data:`GatewayConfig.peers` against the resolved
      service-name set; a peer that references a forge synthesized
      from a project fork URL (rather than from an explicit
      ``forges[]`` entry) is accepted, while a peer that references
      a service name nothing in this config produces is rejected
      with a clear error.

    Returns ``(forges, projects)`` where ``forges`` includes both
    explicitly-declared and synthesized entries.
    """
    forge_specs = list(config.forges)
    forges_by_host = _index_forges_by_host(forge_specs)
    forges_by_name: dict[str, ForgeSpec] = {fs.name: fs for fs in forge_specs}

    resolved_projects: list[ResolvedProject] = []

    for project in config.projects:
        resolved_forks: list[ResolvedFork] = []
        forks = project.resolved_forks()
        for index, fork in enumerate(forks):
            fork_host = _host_of(fork.url)
            if not fork_host:
                raise ValueError(
                    f"Project {project.name!r} fork URL {fork.url!r} "
                    "has no hostname."
                )

            forge_spec: ForgeSpec | None
            if fork.forge:
                forge_spec = forges_by_name.get(fork.forge)
                if forge_spec is None:
                    raise ValueError(
                        f"Project {project.name!r} fork {fork.url!r} "
                        f"references unknown forge {fork.forge!r}.  "
                        f"Known forges: {sorted(forges_by_name)}"
                    )
            else:
                forge_spec = forges_by_host.get(fork_host)
                if forge_spec is None:
                    forge_spec = _synthesize_forge_for_host(
                        fork.url, fork_host, project.name,
                        forges_by_name,
                    )
                    forge_specs.append(forge_spec)
                    forges_by_host[fork_host] = forge_spec
                    forges_by_name[forge_spec.name] = forge_spec

            location = parse_fork_url(forge_spec.type, fork.url)
            fork_name = fork.name or _default_fork_name(
                forge_spec.name,
                is_only_fork=(len(forks) == 1),
            )

            resolved_forks.append(
                ResolvedFork(
                    forge_name=forge_spec.name,
                    forge_type=forge_spec.type,
                    name=fork_name,
                    native_id=fork.native_id or location.native_id,
                    clone_url=location.clone_url,
                    default_branch=fork.default_branch,
                )
            )

        resolved_projects.append(
            ResolvedProject(
                name=project.name,
                default_branch=project.default_branch,
                forks=resolved_forks,
            )
        )

    # Cross-config peer validation runs against the canonical
    # service-name set, not against ``config.forges`` directly.
    # See :func:`validate_peers_against_services` for why this
    # belongs here rather than in the pydantic post-validator on
    # :class:`GatewayConfig`.  Future service categories will join
    # the ``service_names`` set as they grow peer-reference
    # semantics; for now forges are the only category that takes
    # peer accounts.
    service_names = {fs.name for fs in forge_specs}
    service_types_by_name = {fs.name: fs.type for fs in forge_specs}
    validate_peers_against_services(
        list(config.peers),
        service_names,
        service_types_by_name=service_types_by_name,
        account_id_policy=peer_account_id_policy,
    )

    return forge_specs, resolved_projects


def _synthesize_forge_for_host(
    fork_url: str,
    host: str,
    project_name: str,
    forges_by_name: dict[str, ForgeSpec],
) -> ForgeSpec:
    """Build a :class:`ForgeSpec` for *host* on demand.

    Used when a fork URL points at a host that the operator did not
    declare in the ``forges:`` block.  Only succeeds for well-known
    hosts (github.com, gitlab.com) where the forge type is
    unambiguous; otherwise asks the operator to add an explicit entry.
    """
    forge_type = derive_forge_type_from_url(fork_url)
    if forge_type is None:
        raise ValueError(
            f"Project {project_name!r} fork URL {fork_url!r} points at "
            f"host {host!r}, which is not a well-known forge.  Add a "
            "matching entry to the `forges:` array of gateway.json so "
            "its type is unambiguous."
        )
    instance_url = f"https://{host}"
    derived_name = derive_forge_name_from_url(instance_url)
    if derived_name in forges_by_name:
        # An entry with this name already exists but for a different
        # host (we wouldn't have reached here otherwise).  Surface a
        # clear error rather than silently shadowing it.
        raise ValueError(
            f"Auto-synthesized forge name {derived_name!r} for host "
            f"{host!r} collides with an explicit forge entry that "
            f"has URL {forges_by_name[derived_name].url!r}.  Give the "
            "explicit entry a different name."
        )
    return ForgeSpec(
        url=instance_url,
        type=forge_type,
        name=derived_name,
    )


def _default_fork_name(forge_name: str, *, is_only_fork: bool) -> str:
    """Pick a default git remote name for a fork.

    Single-fork projects get ``"origin"`` (matching git's own
    convention).  Multi-fork projects use the forge name as the
    discriminator, since users typically distinguish forks by which
    forge they live on.
    """
    if is_only_fork:
        return "origin"
    return forge_name


# ---------------------------------------------------------------------------
# Service type registry
# ---------------------------------------------------------------------------


SpecToConfigDict = Callable[[BaseModel], dict[str, Any]]
"""Callable that maps a typed-array spec model into kwargs for the
service's ``Config`` model.  See :class:`ServiceTypeRegistry`.
"""


class ServiceTypeRegistry:
    """Maps ``(category, type_key)`` -> service constructor.

    A "category" groups :class:`Service` subclasses that share a typed
    array in ``gateway.json`` -- for example, all forge backends share
    the ``forges:`` array under category ``"forge"``.

    Registration pairs a :class:`Service` subclass with the Pydantic
    ``Config`` model that gets instantiated for it, plus a
    ``spec_to_config`` callable that translates a typed-array entry
    (e.g. a :class:`ForgeSpec`) into kwargs for that ``Config`` model.
    The translation step is per-type because different forge backends
    expose different config shapes (``GitHubConnectionConfig`` carries
    a discriminated ``auth`` block, ``GitLabForgeServiceConfig`` carries
    ``url``/``token``); pushing the translation into the registration
    keeps :func:`instantiate_services` free of any per-type dispatch.
    """

    def __init__(self) -> None:
        self._entries: dict[
            tuple[str, str],
            tuple[type[Service], type[BaseModel], SpecToConfigDict],
        ] = {}

    def register(
        self,
        category: str,
        type_key: str,
        service_cls: type[Service],
        config_cls: type[BaseModel],
        *,
        spec_to_config: SpecToConfigDict,
    ) -> None:
        """Register a service backend under ``(category, type_key)``.

        Subsequent registrations with the same key replace the prior
        entry (which keeps tests and customisation simple).
        """
        self._entries[(category, type_key)] = (
            service_cls, config_cls, spec_to_config,
        )

    def known_types(self, category: str) -> list[str]:
        """Return the registered type keys for *category*, sorted."""
        return sorted(k for c, k in self._entries if c == category)

    def instantiate(
        self,
        category: str,
        type_key: str,
        *,
        spec: BaseModel,
        name: str,
    ) -> Service:
        """Build a service instance for the given typed-array entry.

        Raises :class:`ValueError` when *type_key* is not registered
        under *category*.
        """
        entry = self._entries.get((category, type_key))
        if entry is None:
            raise ValueError(
                f"Unknown {category} type {type_key!r} for entry {name!r}. "
                f"Known {category} types: {self.known_types(category)}"
            )
        service_cls, config_cls, spec_to_config = entry
        config = config_cls(**spec_to_config(spec))
        return service_cls(config, service_name=name)


_REGISTRY: ServiceTypeRegistry | None = None


def get_service_type_registry() -> ServiceTypeRegistry:
    """Return the process-wide :class:`ServiceTypeRegistry` (lazy init).

    Built-in registrations are added on first access; this avoids
    import cycles between :mod:`thorn.gateway._config`,
    :mod:`thorn.tools.forge`, and the forge service modules.
    """
    global _REGISTRY  # noqa: PLW0603
    if _REGISTRY is not None:
        return _REGISTRY
    _REGISTRY = ServiceTypeRegistry()
    _register_builtin_forges(_REGISTRY)
    return _REGISTRY


def _gitlab_forge_spec_to_config(spec: BaseModel) -> dict[str, Any]:
    """Translate a :class:`ForgeSpec` for a GitLab forge into config kwargs."""
    assert isinstance(spec, ForgeSpec)
    if not spec.api_url:
        # Should not normally happen because the model validator fills
        # api_url from url; surface a clear error if it does.
        raise ValueError(
            f"GitLab forge entry {spec.name!r} has no API URL.  Set "
            "`url` (e.g. 'https://gitlab.example.com') in the forge spec."
        )
    return {"url": spec.api_url, "token": ""}


def _github_forge_spec_to_config(spec: BaseModel) -> dict[str, Any]:
    """Translate a :class:`ForgeSpec` for a GitHub forge into config kwargs.

    The token is intentionally empty; per-agent credentials come from
    :class:`~thorn.core._account.ForgeAccountConfig` at the call site
    (see :meth:`ForgeHostService.authenticated_client`).
    """
    assert isinstance(spec, ForgeSpec)
    return {
        "base_url": spec.api_url,
        "auth": {"kind": "pat", "token": ""},
    }


def _register_builtin_forges(registry: ServiceTypeRegistry) -> None:
    """Register the built-in ``"forge"`` backends."""
    from thorn.tools._github_connection import GitHubConnectionConfig
    from thorn.tools.forge import (
        GitHubForgeService,
        GitLabForgeService,
        GitLabForgeServiceConfig,
    )

    registry.register(
        "forge", "gitlab",
        GitLabForgeService, GitLabForgeServiceConfig,
        spec_to_config=_gitlab_forge_spec_to_config,
    )
    registry.register(
        "forge", "github",
        GitHubForgeService, GitHubConnectionConfig,
        spec_to_config=_github_forge_spec_to_config,
    )


# ---------------------------------------------------------------------------
# Loading & instantiation
# ---------------------------------------------------------------------------


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )

        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _describe_supported_agency_config_filenames() -> str:
    return ", ".join(AGENCY_CONFIG_FILENAMES)


def _agency_config_file_format(path: Path) -> AgencyConfigFileFormat:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return AgencyConfigFileFormat.JSON
    if suffix == ".yaml":
        return AgencyConfigFileFormat.YAML
    raise ValueError(f"Unsupported agency configuration filename: {path.name}")


def _discover_agency_config_file(agency_home: Path) -> AgencyConfigFile:
    candidates = [agency_home / name for name in AGENCY_CONFIG_FILENAMES]
    matches = [path for path in candidates if path.is_file()]
    supported_names = _describe_supported_agency_config_filenames()

    if not matches:
        raise FileNotFoundError(
            f"Agency configuration file not found in {agency_home}.\n"
            f"Supported filenames: {supported_names}.\n"
            "Run 'thorn serve bootstrap' to create one, or write it manually."
        )

    if len(matches) > 1:
        found_names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Multiple agency configuration files found in {agency_home}: "
            f"{found_names}.\nKeep exactly one of: {supported_names}."
        )

    path = matches[0]
    return AgencyConfigFile(
        path=path,
        configuration_format=_agency_config_file_format(path),
    )


def _parse_agency_config_file(config_file: AgencyConfigFile) -> dict[str, Any]:
    path = config_file.path
    text = path.read_text(encoding="utf-8")
    raw: Any

    if config_file.configuration_format is AgencyConfigFileFormat.JSON:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    elif config_file.configuration_format is AgencyConfigFileFormat.YAML:
        try:
            raw = yaml.load(text, Loader=_DuplicateKeySafeLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"Malformed YAML in {path}: {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported agency configuration file format: "
            f"{config_file.configuration_format.value}"
        )

    if not isinstance(raw, Mapping):
        raw_type = type(raw).__name__
        raise ValueError(
            f"Agency configuration file {path} must contain a top-level "
            f"mapping, got {raw_type}."
        )
    non_string_keys = [key for key in raw if not isinstance(key, str)]
    if non_string_keys:
        key = non_string_keys[0]
        raise ValueError(
            f"Agency configuration file {path} must use string top-level "
            f"keys, got {type(key).__name__} key {key!r}."
        )
    return dict(raw)


def load_gateway_config(agency_home: Path) -> GatewayConfig:
    """Load and parse the agency config from the given agency home directory.

    *agency_home* is the agency's home root.  It must contain exactly
    one supported agency config filename: ``agency.yaml``,
    ``agency.json``, ``gateway.yaml``, or ``gateway.json``.  Raises
    :class:`FileNotFoundError` when none exist and :class:`ValueError`
    when multiple candidates exist or the file content is malformed.

    No environment-variable expansion happens here: credentials are
    never literal in the config files.  The broker block holds an
    explicit ``admin_api_key_env_var`` field naming the variable;
    agent account credentials carry an explicit ``env_var_name``
    on each entry.  Structural fields (URLs, paths, names) are
    written literally and the framework deliberately keeps the
    env-var convention scoped to credential-bearing blocks to avoid
    surprise for operators with
    literal ``$``-prefixed strings elsewhere.
    """
    config_file = _discover_agency_config_file(agency_home)
    raw = _parse_agency_config_file(config_file)
    return GatewayConfig.model_validate(raw)


def instantiate_services(
    config: GatewayConfig,
    *,
    peer_account_id_policy: PeerAccountIDPolicy = (
        PeerAccountIDPolicy.REJECT_RESOLVABLE_HANDLES
    ),
) -> list[Service]:
    """Create :class:`Service` instances from a gateway configuration.

    Synthesizes any missing forge entries from project URLs (well-known
    hosts only), instantiates each :class:`~thorn.tools.forge.ForgeHostService`
    via the :class:`ServiceTypeRegistry`, and creates one
    :class:`~thorn.tools.forge.ProjectService` per project with
    forks tied to their resolved forges.

    Event sources are **not** created here -- see
    :func:`infer_event_sources`.
    """
    from thorn.tools.forge import (
        ForkConfig,
        ProjectService,
        ProjectServiceConfig,
    )

    registry = get_service_type_registry()

    forge_specs, resolved_projects = _resolve_forges_and_projects(
        config,
        peer_account_id_policy=peer_account_id_policy,
    )

    services: list[Service] = []

    for forge_spec in forge_specs:
        service = registry.instantiate(
            "forge", forge_spec.type,
            spec=forge_spec, name=forge_spec.name,
        )
        log.info(
            "Instantiated %s %r (type=%s, url=%s)",
            type(service).__name__, forge_spec.name,
            forge_spec.type, forge_spec.url,
        )
        services.append(service)

    for project in resolved_projects:
        fork_configs = [
            ForkConfig(
                forge=f.forge_name,
                native_id=f.native_id,
                name=f.name,
                clone_url=f.clone_url,
                default_branch=f.default_branch,
            )
            for f in project.forks
        ]
        proj_cfg = ProjectServiceConfig(
            forks=fork_configs,
            default_branch=project.default_branch,
        )
        proj_svc = ProjectService(proj_cfg, service_name=project.name)
        primary_forge = project.forks[0].forge_name if project.forks else ""
        log.info(
            "Instantiated ProjectService %r (forge=%s)",
            project.name, primary_forge,
        )
        services.append(proj_svc)

    return services


# ---------------------------------------------------------------------------
# Event source inference
# ---------------------------------------------------------------------------


class _ForgeProjectInfo:
    """Per-forge project information used during event source inference."""

    __slots__ = ("repositories", "native_id_to_project_name")

    def __init__(self) -> None:
        self.repositories: list[str] = []
        self.native_id_to_project_name: dict[str, str] = {}


def infer_event_sources(
    config: GatewayConfig,
    agents: list[Any],
) -> list[EventSource]:
    """Create event sources for each (agent, forge) pair.

    For each agent that has a :class:`ForgeAccountConfig` on a forge
    declared in *config*, an appropriate event source is created:

    - **GitHub**: A notifications poller authenticated with the
      agent's PAT (notifications are user-scoped, no per-repo
      enumeration needed).
    - **GitLab**: A TODOs poller authenticated with the agent's
      credentials (TODOs are user-scoped, no per-repo enumeration).

    The ``poll_interval`` comes from the forge spec in the config.
    Project names are threaded through to the event sources so that
    session keys use project-name-based routing.
    """
    from thorn.core._account import AgentAccountsConfig

    forge_specs, resolved_projects = _resolve_forges_and_projects(config)
    forge_specs_by_name: dict[str, ForgeSpec] = {
        f.name: f for f in forge_specs
    }

    forge_project_info: dict[str, _ForgeProjectInfo] = {}
    for proj in resolved_projects:
        for fork in proj.forks:
            info = forge_project_info.setdefault(
                fork.forge_name, _ForgeProjectInfo(),
            )
            info.repositories.append(fork.native_id)
            info.native_id_to_project_name[fork.native_id] = proj.name

    sources: list[EventSource] = []

    for agent in agents:
        accounts: AgentAccountsConfig | None = getattr(agent, "accounts", None)
        if accounts is None:
            continue

        forge_accounts = [
            acct
            for acct in accounts.accounts
            if getattr(acct, "service", None) in forge_specs_by_name
        ]
        if not forge_accounts:
            continue

        owner_agent_id = _event_source_owner_agent_id(agent)
        if owner_agent_id is None:
            agent_name = (
                getattr(agent, "name", None)
                or getattr(agent, "id", "unknown")
            )
            log.warning(
                "Skipping inferred event sources for agent %r: agent "
                "has no persistent id.",
                agent_name,
            )
            continue

        for acct in forge_accounts:
            forge_spec = forge_specs_by_name[acct.service]
            info = forge_project_info.get(forge_spec.name, _ForgeProjectInfo())
            source = _create_event_source_for_account(
                forge_spec=forge_spec,
                account=acct,
                agent=agent,
                owner_agent_id=owner_agent_id,
                native_id_to_project_name=info.native_id_to_project_name,
            )
            if source is not None:
                sources.append(source)

    return sources


def _event_source_owner_agent_id(agent: Any) -> AgentID | None:
    """Return the persistent agent ID that owns inferred account sources."""
    agent_id = getattr(agent, "id", None)
    if isinstance(agent_id, AgentID):
        return agent_id
    if agent_id is None:
        return None
    agent_id_text = str(agent_id).strip()
    if not agent_id_text:
        return None
    return AgentID(agent_id_text)


def _create_event_source_for_account(
    *,
    forge_spec: ForgeSpec,
    account: Any,
    agent: Any,
    owner_agent_id: AgentID,
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a single event source for an agent's account on a forge."""
    agent_name = getattr(agent, "name", None) or getattr(agent, "id", "unknown")

    if forge_spec.type == "github":
        return _create_github_source(
            forge_spec=forge_spec,
            account=account,
            agent_name=str(agent_name),
            owner_agent_id=owner_agent_id,
            native_id_to_project_name=native_id_to_project_name,
        )

    if forge_spec.type == "gitlab":
        return _create_gitlab_source(
            forge_spec=forge_spec,
            account=account,
            agent_name=str(agent_name),
            owner_agent_id=owner_agent_id,
            native_id_to_project_name=native_id_to_project_name,
        )

    log.warning(
        "No event source strategy for forge type %r (forge=%r, agent=%r)",
        forge_spec.type, forge_spec.name, agent_name,
    )
    return None


def _resolve_event_source_token(
    account: Any,
    *,
    kind: str,
    forge_spec: ForgeSpec,
    agent_name: str,
) -> str | None:
    """Look up and read the event-source credential off *account*.

    Returns ``None`` (with a warning logged) when the account has
    no credential of the requested *kind*, or when the env var
    referenced by the credential is not set.  Centralising this
    keeps the per-forge source helpers free of the same
    walk-the-credentials boilerplate.
    """
    from thorn.core._account import find_credential
    from thorn.core._credentials import CredentialMissingError

    cred = find_credential(account, kind=kind)
    if cred is None:
        log.warning(
            "Skipping event source for forge %r (agent=%r): account "
            "has no credential of kind %r.",
            forge_spec.name, agent_name, kind,
        )
        return None
    try:
        return str(cred.read_value())
    except CredentialMissingError as exc:
        log.warning(
            "Skipping event source for forge %r (agent=%r): %s",
            forge_spec.name, agent_name, exc,
        )
        return None


def _create_github_source(
    *,
    forge_spec: ForgeSpec,
    account: Any,
    agent_name: str,
    owner_agent_id: AgentID,
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a GitHub notifications source for one (agent, forge) pair.

    The Notifications API is user-scoped (like GitLab TODOs), so no
    repository list is needed.  Only ``"pat"``-kind credentials are
    used; the GitHub App auth flow is not supported here.
    """
    token = _resolve_event_source_token(
        account, kind="pat",
        forge_spec=forge_spec, agent_name=agent_name,
    )
    if token is None:
        return None

    from thorn.gateway.sources._github import (
        GitHubNotificationsSource,
        GitHubNotificationsSourceConfig,
    )

    source_name = f"{agent_name}-{forge_spec.name}-events"

    cfg = GitHubNotificationsSourceConfig(
        token=token,
        base_url=forge_spec.api_url,
        poll_interval=forge_spec.poll_interval,
        native_id_to_project_name=native_id_to_project_name,
        forge_name=forge_spec.name,
    )
    source = GitHubNotificationsSource(
        cfg,
        service_name=source_name,
        owner_agent_id=owner_agent_id,
    )
    log.info(
        "Inferred GitHub notifications source %r (agent=%s)",
        source_name, agent_name,
    )
    return source


def _create_gitlab_source(
    *,
    forge_spec: ForgeSpec,
    account: Any,
    agent_name: str,
    owner_agent_id: AgentID,
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a GitLab TODOs source for one (agent, forge) pair.

    GitLab TODOs are user-scoped, so no repository list is needed.
    The *native_id_to_project_name* mapping (keyed by the GitLab
    project's path-with-namespace) is passed through to the source so
    that session keys use project-name-based routing.
    """
    token = _resolve_event_source_token(
        account, kind="gitlab-pat",
        forge_spec=forge_spec, agent_name=agent_name,
    )
    if token is None:
        return None

    from thorn.gateway.sources._gitlab import GitLabSourceConfig, GitLabTODOsSource

    source_name = f"{agent_name}-{forge_spec.name}-events"
    cfg = GitLabSourceConfig(
        url=forge_spec.api_url,
        token=token,
        poll_interval=forge_spec.poll_interval,
        project_id_to_name=native_id_to_project_name,
        forge_name=forge_spec.name,
    )
    source = GitLabTODOsSource(
        cfg,
        service_name=source_name,
        owner_agent_id=owner_agent_id,
    )
    log.info(
        "Inferred GitLab event source %r (agent=%s)",
        source_name, agent_name,
    )
    return source


__all__ = [
    "AgentSandboxOverride",
    "BrokerConfig",
    "BundledBrokerImageConfig",
    "ForgeSpec",
    "ForkLocation",
    "ForkSpec",
    "GATEWAY_CONFIG_FILENAME",
    "GatewayConfig",
    "OCIImageReference",
    "PeerAccountHandleOnlyProblem",
    "PeerAccountIDPolicy",
    "PlannedEgressAllowlistEntry",
    "ProjectSpec",
    "ResolvedFork",
    "ResolvedProject",
    "SandboxConfig",
    "ServiceTypeRegistry",
    "THORN_BUNDLED_BROKER_ONECLI_IMAGE_ENV_VAR",
    "THORN_BUNDLED_BROKER_POSTGRES_IMAGE_ENV_VAR",
    "THORN_BUNDLED_BROKER_PROXY_IMAGE_ENV_VAR",
    "derive_api_url",
    "derive_forge_name_from_url",
    "derive_forge_type_from_url",
    "expand_env_vars",
    "get_service_type_registry",
    "infer_event_sources",
    "instantiate_services",
    "load_gateway_config",
    "parse_fork_url",
    "collect_handle_only_peer_account_problems",
    "validate_peers_against_services",
]
