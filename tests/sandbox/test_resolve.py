"""Tests for :func:`resolve_sandbox_config` (agency + agent merge)."""

from __future__ import annotations

from thorn.gateway._config import AgentSandboxOverride, SandboxConfig
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
