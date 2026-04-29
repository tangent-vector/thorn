"""Merge agency-wide sandbox defaults with per-agent overrides.

Phase B's configuration story is two-tier: the agency-wide
:class:`~thorn.gateway._config.SandboxConfig` block in
``gateway.json`` sets defaults for every agent in the agency, and the
optional :class:`~thorn.gateway._config.AgentSandboxOverride` block
in ``agent.json`` lets a single agent diverge.  This module is the
single place where the two are merged into a single
:class:`ResolvedSandboxConfig` ready for the runtime to act on.

Centralising the merge here means the rules are auditable in one
place: which fields are "override wins" and which are "additive
union" lives in :func:`resolve_sandbox_config`, not scattered through
the call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from thorn.gateway._config import (
    AgentSandboxOverride,
    EgressAllowlistEntry,
    SandboxConfig,
)
from thorn.sandbox._image import default_sandbox_image_tag


@dataclass(frozen=True)
class ResolvedSandboxConfig:
    """Per-agent sandbox configuration after agency + override merge.

    All fields carry their *effective* value -- no nullable shape
    survives the merge.  Callers do not need to look at either the
    agency or the agent block again.

    Phase E hardening fields (``capabilities_drop``,
    ``capabilities_add``, ``security_opts``, ``read_only_root``,
    ``memory_limit``, ``cpu_limit``, ``pid_limit``) are present on
    every resolved config so the runtime can populate the matching
    :class:`~thorn.sandbox._container.ContainerHostConfig` fields
    uniformly without having to re-derive defaults at the use site.
    """

    backend: Literal["subprocess", "container"]
    oci_runtime: Literal["podman", "docker"] | None
    image: str
    env_passthrough: tuple[str, ...]
    extra_env: tuple[tuple[str, str], ...]
    dev_mount_runtime: bool
    container_ready_timeout_s: float
    egress_network: str | None
    egress_allowlist: tuple[EgressAllowlistEntry, ...]

    # Phase E hardening, all populated.
    capabilities_drop: tuple[str, ...]
    capabilities_add: tuple[str, ...]
    security_opts: tuple[str, ...]
    read_only_root: bool
    memory_limit: str | None
    cpu_limit: float | None
    pid_limit: int | None


_DEFAULT_AGENCY_SANDBOX = SandboxConfig(backend="subprocess")
"""Used when ``gateway.json`` omits the ``sandbox`` block entirely.

Phase B's stance is that an agency must explicitly opt into the
container backend by writing the block, so the implicit default is
the Phase-A subprocess executor.  This keeps existing agencies that
have not been told about Phase B running unchanged.
"""


def _additive_str_list(
    agency_values: list[str] | tuple[str, ...],
    override_values: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Concatenate two string lists, preserving order and dropping duplicates.

    First occurrence wins on ordering: agency values come first in
    the order they were declared, then any override values that did
    not already appear.  This mirrors the established
    ``env_passthrough`` merge rule and gives operators a single
    consistent mental model for "list of strings" fields across
    Phase B (env_passthrough), Phase E (capabilities_drop / add,
    security_opts), and any future additions.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for value in agency_values:
        if value not in seen:
            merged.append(value)
            seen.add(value)
    if override_values is not None:
        for value in override_values:
            if value not in seen:
                merged.append(value)
                seen.add(value)
    return tuple(merged)


def resolve_sandbox_config(
    agency: SandboxConfig | None,
    override: AgentSandboxOverride | None,
) -> ResolvedSandboxConfig:
    """Merge agency-wide and per-agent sandbox configuration.

    Rules per field:

    * ``backend``: agent overrides agency overrides subprocess
      default.
    * ``oci_runtime``: agency-only (no per-agent override; an agency
      runs one runtime).
    * ``image``: agent (when non-empty) overrides agency (when
      non-empty) overrides the framework default
      :func:`default_sandbox_image_tag`.
    * ``env_passthrough``: *additive* -- the agent's list is
      concatenated to the agency's, with duplicates removed while
      preserving first-occurrence order.  Per the Phase-B plan,
      per-agent config can broaden the allow-list but not narrow it.
    * ``extra_env``: agent-only (literal env entries; the agency
      block has no equivalent because forwarding non-secret literal
      env entries agency-wide is rare and we'd rather force
      operators to put it on a single agent's block where the intent
      is obvious).
    * ``dev_mount_runtime``: agency-only (it is a developer toggle
      that should apply uniformly across an agency).
    * ``container_ready_timeout_s``: agent overrides agency overrides
      30s default.
    * ``egress_network``, ``egress_allowlist``: agency-only at
      this writing.  See the Phase D retro for rationale.
    * Phase E hardening lists (``capabilities_drop``,
      ``capabilities_add``, ``security_opts``): *additive*, same
      semantics as ``env_passthrough``.  An empty per-agent list
      means "no addition" rather than "reset agency to nothing".
    * Phase E hardening scalars (``read_only_root``,
      ``memory_limit``, ``cpu_limit``, ``pid_limit``): per-agent
      value replaces the agency value when the override is set
      (``None`` for the override means "use agency").
    """
    a = agency if agency is not None else _DEFAULT_AGENCY_SANDBOX

    backend: Literal["subprocess", "container"] = a.backend
    if override is not None and override.backend is not None:
        backend = override.backend

    image = ""
    if override is not None and override.image:
        image = override.image
    elif a.image:
        image = a.image
    else:
        image = default_sandbox_image_tag()

    env_passthrough = _additive_str_list(
        a.env_passthrough,
        override.env_passthrough if override is not None else None,
    )

    extra_env: tuple[tuple[str, str], ...] = ()
    if override is not None and override.extra_env:
        extra_env = tuple(override.extra_env.items())

    timeout = a.container_ready_timeout_s
    if override is not None and override.container_ready_timeout_s is not None:
        timeout = override.container_ready_timeout_s

    capabilities_drop = _additive_str_list(
        a.capabilities_drop,
        override.capabilities_drop if override is not None else None,
    )
    capabilities_add = _additive_str_list(
        a.capabilities_add,
        override.capabilities_add if override is not None else None,
    )
    security_opts = _additive_str_list(
        a.security_opts,
        override.security_opts if override is not None else None,
    )

    read_only_root = a.read_only_root
    if override is not None and override.read_only_root is not None:
        read_only_root = override.read_only_root

    memory_limit = a.memory_limit
    if override is not None and override.memory_limit is not None:
        memory_limit = override.memory_limit

    cpu_limit = a.cpu_limit
    if override is not None and override.cpu_limit is not None:
        cpu_limit = override.cpu_limit

    pid_limit = a.pid_limit
    if override is not None and override.pid_limit is not None:
        pid_limit = override.pid_limit

    return ResolvedSandboxConfig(
        backend=backend,
        oci_runtime=a.oci_runtime,
        image=image,
        env_passthrough=env_passthrough,
        extra_env=extra_env,
        dev_mount_runtime=a.dev_mount_runtime,
        container_ready_timeout_s=timeout,
        # Phase D: egress fields are agency-only.  No per-agent
        # override surface today: the network and allow-list are
        # operator-controlled because letting an agent narrow or
        # widen them would defeat the broker-only invariant.  If a
        # specialised agent ever needs different egress, the right
        # mechanism is a separate agency, not a per-agent escape
        # hatch.
        egress_network=a.egress_network,
        egress_allowlist=tuple(a.egress_allowlist),
        capabilities_drop=capabilities_drop,
        capabilities_add=capabilities_add,
        security_opts=security_opts,
        read_only_root=read_only_root,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        pid_limit=pid_limit,
    )


__all__ = [
    "ResolvedSandboxConfig",
    "resolve_sandbox_config",
]
