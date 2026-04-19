"""Gateway configuration: loading services from an agency's ``gateway.json``.

The gateway configuration file lives in the *agency home* directory
(``<agency_home>/gateway.json``) and declares the agency's workspace
directory along with its forges and projects.  Forges are external
platforms that host version-controlled repositories; projects are
logical software projects with one or more forks hosted on those
forges.

The on-disk format uses a top-level ``"workspace"`` string and typed
arrays.  In the simplest case, the user only writes ``"projects"``
and the rest is inferred::

    {
      "workspace": "/home/me/thorn-workspace",
      "projects": [
        {
          "name": "tiny-talk",
          "url": "https://github.com/tangent-vector/tiny-talk"
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
path is resolved against the directory containing ``gateway.json``
(the agency home) -- see :meth:`GatewayConfig.resolve_workspace`.

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

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from thorn.core._service import Service
from thorn.gateway._event import EventSource

log = logging.getLogger(__name__)

GATEWAY_CONFIG_FILENAME = "gateway.json"


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
    ``"gitlab-master-nvidia-com"``.

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
# $ENV_VAR expansion
# ---------------------------------------------------------------------------


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
        return [ForkSpec(url=self.url)]


class GatewayConfig(BaseModel):
    """Top-level model for an agency's ``gateway.json``.

    Carries the agency's workspace directory along with its declared
    forges and projects.  Future plug-in service categories will
    appear as additional typed array fields (for example,
    ``messaging_services: list[...]``).
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


def _resolve_forges_and_projects(
    config: GatewayConfig,
) -> tuple[list[ForgeSpec], list[ResolvedProject]]:
    """Resolve forge entries and projects for *config*.

    - Walks each project's forks to ensure a matching :class:`ForgeSpec`
      exists; synthesizes one for any well-known host that lacks an
      explicit entry.
    - Replaces each :class:`ForkSpec` with a :class:`ResolvedFork` whose
      ``forge``, ``name``, ``native_id``, and ``clone_url`` are filled in.
    - Detects collisions where two different forge specs end up with
      the same ``name``.

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
                    native_id=location.native_id,
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


def load_gateway_config(agency_home: Path) -> GatewayConfig:
    """Load and parse ``gateway.json`` from the given agency home directory.

    *agency_home* is the directory containing ``gateway.json`` (the
    agency's home root).  Raises :class:`FileNotFoundError` if the
    config file does not exist.
    """
    config_path = agency_home / GATEWAY_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Gateway configuration file not found: {config_path}\n"
            "Run 'thorn serve bootstrap' to create one, or write it manually."
        )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(raw)


def instantiate_services(config: GatewayConfig) -> list[Service]:
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

    forge_specs, resolved_projects = _resolve_forges_and_projects(config)

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

        for acct in accounts.forge_accounts():
            forge_spec = forge_specs_by_name.get(acct.service)
            if forge_spec is None:
                log.warning(
                    "Agent %r has account on forge %r which is not in gateway config; skipping.",
                    getattr(agent, "name", "?"), acct.service,
                )
                continue

            info = forge_project_info.get(forge_spec.name, _ForgeProjectInfo())
            source = _create_event_source_for_account(
                forge_spec=forge_spec,
                account=acct,
                agent=agent,
                native_id_to_project_name=info.native_id_to_project_name,
            )
            if source is not None:
                sources.append(source)

    return sources


def _create_event_source_for_account(
    *,
    forge_spec: ForgeSpec,
    account: Any,
    agent: Any,
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a single event source for an agent's account on a forge."""
    from thorn.core._account import ForgeAccountConfig

    assert isinstance(account, ForgeAccountConfig)
    agent_name = getattr(agent, "name", None) or getattr(agent, "id", "unknown")

    if forge_spec.type == "github":
        return _create_github_source(
            forge_spec=forge_spec,
            account=account,
            agent_name=str(agent_name),
            native_id_to_project_name=native_id_to_project_name,
        )

    if forge_spec.type == "gitlab":
        return _create_gitlab_source(
            forge_spec=forge_spec,
            account=account,
            agent_name=str(agent_name),
            native_id_to_project_name=native_id_to_project_name,
        )

    log.warning(
        "No event source strategy for forge type %r (forge=%r, agent=%r)",
        forge_spec.type, forge_spec.name, agent_name,
    )
    return None


def _create_github_source(
    *,
    forge_spec: ForgeSpec,
    account: Any,
    agent_name: str,
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a GitHub notifications source for one (agent, forge) pair.

    The Notifications API is user-scoped (like GitLab TODOs), so no
    repository list is needed.  Only PAT credentials are supported;
    GitHub App installation tokens cannot access the Notifications API.
    """
    from thorn.tools._github_connection import GitHubPatAuth

    creds = account.credentials
    if not isinstance(creds, GitHubPatAuth):
        log.warning(
            "GitHub notifications require a PAT; credential type %s "
            "for forge %r is not supported (agent=%r). "
            "GitHub App installation tokens cannot access the Notifications API.",
            type(creds).__name__, forge_spec.name, agent_name,
        )
        return None

    from thorn.gateway.sources._github import (
        GitHubNotificationsSource,
        GitHubNotificationsSourceConfig,
    )

    source_name = f"{agent_name}-{forge_spec.name}-events"

    cfg = GitHubNotificationsSourceConfig(
        token=creds.token,
        base_url=forge_spec.api_url,
        poll_interval=forge_spec.poll_interval,
        native_id_to_project_name=native_id_to_project_name,
    )
    source = GitHubNotificationsSource(cfg, service_name=source_name)
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
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a GitLab TODOs source for one (agent, forge) pair.

    GitLab TODOs are user-scoped, so no repository list is needed.
    The *native_id_to_project_name* mapping (keyed by the GitLab
    project's path-with-namespace) is passed through to the source so
    that session keys use project-name-based routing.
    """
    from thorn.core._account import GitLabCredentials

    creds = account.credentials
    if not isinstance(creds, GitLabCredentials):
        log.warning(
            "Unsupported credential type %s for GitLab forge %r",
            type(creds).__name__, forge_spec.name,
        )
        return None

    from thorn.gateway.sources._gitlab import GitLabSourceConfig, GitLabTODOsSource

    source_name = f"{agent_name}-{forge_spec.name}-events"
    # GitLab source uses the instance URL (python-gitlab adds /api/v4
    # internally).  api_url and url collapse to the same value for
    # GitLab in the resolved spec.
    cfg = GitLabSourceConfig(
        url=forge_spec.api_url,
        token=creds.token,
        poll_interval=forge_spec.poll_interval,
        project_id_to_name=native_id_to_project_name,
    )
    source = GitLabTODOsSource(cfg, service_name=source_name)
    log.info(
        "Inferred GitLab event source %r (agent=%s)",
        source_name, agent_name,
    )
    return source


__all__ = [
    "ForgeSpec",
    "ForkLocation",
    "ForkSpec",
    "GATEWAY_CONFIG_FILENAME",
    "GatewayConfig",
    "ProjectSpec",
    "ResolvedFork",
    "ResolvedProject",
    "ServiceTypeRegistry",
    "derive_api_url",
    "derive_forge_name_from_url",
    "derive_forge_type_from_url",
    "expand_env_vars",
    "get_service_type_registry",
    "infer_event_sources",
    "instantiate_services",
    "load_gateway_config",
    "parse_fork_url",
]
