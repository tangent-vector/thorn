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


_DEFAULT_AGENCY_SANDBOX = SandboxConfig(backend="subprocess")
"""Used when ``gateway.json`` omits the ``sandbox`` block entirely.

Phase B's stance is that an agency must explicitly opt into the
container backend by writing the block, so the implicit default is
the Phase-A subprocess executor.  This keeps existing agencies that
have not been told about Phase B running unchanged.
"""


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

    merged_passthrough: list[str] = []
    seen: set[str] = set()
    for name in a.env_passthrough:
        if name not in seen:
            merged_passthrough.append(name)
            seen.add(name)
    if override is not None:
        for name in override.env_passthrough:
            if name not in seen:
                merged_passthrough.append(name)
                seen.add(name)

    extra_env: tuple[tuple[str, str], ...] = ()
    if override is not None and override.extra_env:
        extra_env = tuple(override.extra_env.items())

    timeout = a.container_ready_timeout_s
    if override is not None and override.container_ready_timeout_s is not None:
        timeout = override.container_ready_timeout_s

    return ResolvedSandboxConfig(
        backend=backend,
        oci_runtime=a.oci_runtime,
        image=image,
        env_passthrough=tuple(merged_passthrough),
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
    )


__all__ = [
    "ResolvedSandboxConfig",
    "resolve_sandbox_config",
]
