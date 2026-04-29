"""Tests for :func:`resolve_sandbox_config` (agency + agent merge)."""

from __future__ import annotations

from thorn.gateway._config import (
    AgentSandboxOverride,
    EgressAllowlistEntry,
    SandboxConfig,
)
from thorn.sandbox import default_sandbox_image_tag, resolve_sandbox_config


class TestNoConfig:
    def test_no_agency_no_override_defaults_to_subprocess(self) -> None:
        resolved = resolve_sandbox_config(None, None)
        assert resolved.backend == "subprocess"
        assert resolved.image == default_sandbox_image_tag()
        assert resolved.env_passthrough == ()
        assert resolved.extra_env == ()
        assert resolved.dev_mount_runtime is False
        assert resolved.oci_runtime is None


class TestAgencyOnly:
    def test_agency_block_picks_container_default(self) -> None:
        agency = SandboxConfig()
        resolved = resolve_sandbox_config(agency, None)
        assert resolved.backend == "container"

    def test_agency_image_propagates(self) -> None:
        agency = SandboxConfig(image="thorn-sandbox:0.5")
        resolved = resolve_sandbox_config(agency, None)
        assert resolved.image == "thorn-sandbox:0.5"

    def test_empty_agency_image_falls_back_to_default(self) -> None:
        agency = SandboxConfig(image="")
        resolved = resolve_sandbox_config(agency, None)
        assert resolved.image == default_sandbox_image_tag()

    def test_dev_mount_runtime_propagates(self) -> None:
        agency = SandboxConfig(dev_mount_runtime=True)
        assert resolve_sandbox_config(agency, None).dev_mount_runtime is True


class TestOverride:
    def test_override_image_replaces_agency(self) -> None:
        agency = SandboxConfig(image="agency:1")
        override = AgentSandboxOverride(image="agent:2")
        assert resolve_sandbox_config(agency, override).image == "agent:2"

    def test_empty_override_image_does_not_replace(self) -> None:
        agency = SandboxConfig(image="agency:1")
        override = AgentSandboxOverride(image=None)
        assert resolve_sandbox_config(agency, override).image == "agency:1"

    def test_override_backend_wins(self) -> None:
        agency = SandboxConfig(backend="container")
        override = AgentSandboxOverride(backend="subprocess")
        assert resolve_sandbox_config(agency, override).backend == "subprocess"

    def test_env_passthrough_is_additive_and_dedup(self) -> None:
        agency = SandboxConfig(env_passthrough=["LANG", "TZ"])
        override = AgentSandboxOverride(env_passthrough=["TZ", "LC_ALL"])
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.env_passthrough == ("LANG", "TZ", "LC_ALL")

    def test_extra_env_carries_through(self) -> None:
        override = AgentSandboxOverride(
            extra_env={"FOO": "bar", "BAZ": "qux"},
        )
        resolved = resolve_sandbox_config(SandboxConfig(), override)
        assert dict(resolved.extra_env) == {"FOO": "bar", "BAZ": "qux"}

    def test_timeout_override_wins(self) -> None:
        agency = SandboxConfig(container_ready_timeout_s=10.0)
        override = AgentSandboxOverride(container_ready_timeout_s=60.0)
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.container_ready_timeout_s == 60.0

    def test_oci_runtime_is_agency_only(self) -> None:
        # AgentSandboxOverride deliberately has no oci_runtime field;
        # resolve should propagate the agency value either way.
        agency = SandboxConfig(oci_runtime="docker")
        override = AgentSandboxOverride()
        assert resolve_sandbox_config(agency, override).oci_runtime == "docker"


class TestEgressFields:
    """Phase D: ``egress_network`` and ``egress_allowlist`` are
    agency-only.  The resolver carries them through verbatim and
    the per-agent override surface deliberately does not include
    them (operator-controlled invariant -- a per-agent escape hatch
    would defeat the broker-only egress policy)."""

    def test_egress_defaults_are_unset(self) -> None:
        resolved = resolve_sandbox_config(None, None)
        assert resolved.egress_network is None
        assert resolved.egress_allowlist == ()

    def test_egress_network_propagates_from_agency(self) -> None:
        agency = SandboxConfig(egress_network="thorn-broker")
        resolved = resolve_sandbox_config(agency, None)
        assert resolved.egress_network == "thorn-broker"

    def test_egress_allowlist_propagates_as_tuple(self) -> None:
        agency = SandboxConfig(
            egress_allowlist=[
                EgressAllowlistEntry(host="status.internal", port=8080),
                EgressAllowlistEntry(host="metrics.internal", port=443),
            ],
        )
        resolved = resolve_sandbox_config(agency, None)
        assert isinstance(resolved.egress_allowlist, tuple)
        assert [(e.host, e.port) for e in resolved.egress_allowlist] == [
            ("status.internal", 8080),
            ("metrics.internal", 443),
        ]

    def test_egress_fields_unaffected_by_override(self) -> None:
        agency = SandboxConfig(egress_network="thorn-broker")
        override = AgentSandboxOverride()
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.egress_network == "thorn-broker"


class TestPhaseEHardeningDefaults:
    """Phase E adds hardening fields to :class:`SandboxConfig` with
    conservative-by-default values.  These tests pin the defaults so
    a future operator-facing change to "what does an agency get when
    they write ``sandbox: {}``?" cannot land silently."""

    def test_fresh_agency_block_has_conservative_caps(self) -> None:
        resolved = resolve_sandbox_config(SandboxConfig(), None)
        assert resolved.capabilities_drop == ("ALL",)
        assert resolved.capabilities_add == ()

    def test_fresh_agency_block_emits_no_new_privileges(self) -> None:
        resolved = resolve_sandbox_config(SandboxConfig(), None)
        assert "no-new-privileges" in resolved.security_opts

    def test_fresh_agency_block_has_readonly_rootfs(self) -> None:
        resolved = resolve_sandbox_config(SandboxConfig(), None)
        assert resolved.read_only_root is True

    def test_fresh_agency_block_has_resource_limits(self) -> None:
        resolved = resolve_sandbox_config(SandboxConfig(), None)
        assert resolved.memory_limit == "2G"
        assert resolved.cpu_limit == 2.0
        assert resolved.pid_limit == 512

    def test_no_agency_no_override_inherits_defaults(self) -> None:
        # Even when ``gateway.json`` omits the sandbox block entirely
        # (the implicit subprocess-backend path), the resolver still
        # populates Phase-E hardening fields with their defaults.
        # The subprocess backend ignores them; this just keeps the
        # ResolvedSandboxConfig shape uniform across backends so
        # downstream code can read the fields without conditional
        # guards.
        resolved = resolve_sandbox_config(None, None)
        assert resolved.capabilities_drop == ("ALL",)
        assert resolved.read_only_root is True
        assert resolved.memory_limit == "2G"


class TestPhaseEHardeningMergeRules:
    """The Phase-E plan adopts the existing ``env_passthrough`` merge
    rule for every list-typed hardening field (additive, dedup,
    agency-first ordering) and a uniform "scalar replace when set"
    rule for every scalar field.  These tests pin the rules so a
    future config refactor cannot quietly diverge from them."""

    def test_capabilities_drop_is_additive(self) -> None:
        agency = SandboxConfig(capabilities_drop=["ALL"])
        override = AgentSandboxOverride(capabilities_drop=["AUDIT_CONTROL"])
        resolved = resolve_sandbox_config(agency, override)
        # Agency value comes first; override entry appended.
        assert resolved.capabilities_drop == ("ALL", "AUDIT_CONTROL")

    def test_capabilities_drop_dedup_preserves_first(self) -> None:
        agency = SandboxConfig(capabilities_drop=["ALL", "NET_RAW"])
        override = AgentSandboxOverride(capabilities_drop=["NET_RAW", "AUDIT_CONTROL"])
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.capabilities_drop == ("ALL", "NET_RAW", "AUDIT_CONTROL")

    def test_capabilities_add_is_additive(self) -> None:
        agency = SandboxConfig(capabilities_add=["NET_BIND_SERVICE"])
        override = AgentSandboxOverride(capabilities_add=["NET_RAW"])
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.capabilities_add == ("NET_BIND_SERVICE", "NET_RAW")

    def test_security_opts_is_additive(self) -> None:
        agency = SandboxConfig(security_opts=["no-new-privileges"])
        override = AgentSandboxOverride(
            security_opts=["apparmor=my-profile"],
        )
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.security_opts == (
            "no-new-privileges", "apparmor=my-profile",
        )

    def test_empty_override_list_is_no_op(self) -> None:
        # Empty override list means "no addition", not "reset to
        # nothing".  Mirrors the env_passthrough rule.
        agency = SandboxConfig(capabilities_drop=["ALL"])
        override = AgentSandboxOverride(capabilities_drop=[])
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.capabilities_drop == ("ALL",)

    def test_read_only_root_per_agent_can_disable(self) -> None:
        agency = SandboxConfig(read_only_root=True)
        override = AgentSandboxOverride(read_only_root=False)
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.read_only_root is False

    def test_read_only_root_per_agent_can_enable(self) -> None:
        agency = SandboxConfig(read_only_root=False)
        override = AgentSandboxOverride(read_only_root=True)
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.read_only_root is True

    def test_read_only_root_unset_override_inherits_agency(self) -> None:
        agency = SandboxConfig(read_only_root=False)
        override = AgentSandboxOverride()
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.read_only_root is False

    def test_memory_limit_override_replaces(self) -> None:
        agency = SandboxConfig(memory_limit="2G")
        override = AgentSandboxOverride(memory_limit="32G")
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.memory_limit == "32G"

    def test_memory_limit_none_override_inherits_agency(self) -> None:
        agency = SandboxConfig(memory_limit="2G")
        override = AgentSandboxOverride(memory_limit=None)
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.memory_limit == "2G"

    def test_cpu_limit_override_replaces(self) -> None:
        agency = SandboxConfig(cpu_limit=2.0)
        override = AgentSandboxOverride(cpu_limit=8.0)
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.cpu_limit == 8.0

    def test_pid_limit_override_replaces(self) -> None:
        agency = SandboxConfig(pid_limit=512)
        override = AgentSandboxOverride(pid_limit=4096)
        resolved = resolve_sandbox_config(agency, override)
        assert resolved.pid_limit == 4096

    def test_agency_can_remove_resource_caps(self) -> None:
        # ``null`` in the agency JSON removes a default cap entirely;
        # the override path's ``None`` then leaves it removed.
        agency = SandboxConfig(
            memory_limit=None, cpu_limit=None, pid_limit=None,
        )
        resolved = resolve_sandbox_config(agency, None)
        assert resolved.memory_limit is None
        assert resolved.cpu_limit is None
        assert resolved.pid_limit is None
