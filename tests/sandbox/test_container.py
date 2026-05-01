"""Tests for :class:`ContainerDaemonHost` driving the fake OCI adapter.

These tests cover the host's behavior in isolation -- bind-mount
construction, env passthrough, two-stage readiness probe, idempotent
teardown, and image-missing diagnostics.  The real-podman path is
exercised by ``tests/sandbox/test_smoke.py`` (gated on
``@pytest.mark.requires_podman``).

Most tests pre-populate the fake adapter with the relevant image so
the happy path "container reaches running" works without further
scripting; tests that exercise failure modes script ``inspect``
results explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from thorn.sandbox import (
    CONTAINER_CONTROL_DIR,
    CONTAINER_GIT_CONFIG_PATH,
    CONTAINER_HOME_DIR,
    CONTAINER_RUNTIME_DIR,
    CONTAINER_SOCKET_PATH,
    CONTAINER_WORKSPACE_DIR,
    ContainerDaemonHost,
    ContainerHostConfig,
    ContainerNotReadyError,
    ContainerStartTimeoutError,
    ENTRYPOINT_REQUIRED_CAPS,
    FakeOCIRuntimeAdapter,
    SandboxImageMissingError,
    derive_container_name,
)


def _make_config(
    tmp_path: Path,
    *,
    image: str = "thorn-sandbox:test",
    adapter: FakeOCIRuntimeAdapter | None = None,
    dev_mount_runtime: Path | None = None,
    env_passthrough: tuple[str, ...] = (),
    extra_env: tuple[tuple[str, str], ...] = (),
    user: str | None = "1000:1000",
    container_ready_timeout_s: float = 1.0,
    container_ready_poll_s: float = 0.01,
) -> ContainerHostConfig:
    if adapter is None:
        adapter = FakeOCIRuntimeAdapter(present_images=[image])
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    control = tmp_path / "control"
    home.mkdir()
    workspace.mkdir()
    return ContainerHostConfig(
        agent_id="agent-x",
        container_name="thorn-agent-agent-x",
        image=image,
        adapter=adapter,
        host_home_dir=home,
        host_workspace_dir=workspace,
        host_control_dir=control,
        env_passthrough=env_passthrough,
        extra_env=extra_env,
        dev_mount_runtime=dev_mount_runtime,
        user=user,
        container_ready_timeout_s=container_ready_timeout_s,
        container_ready_poll_s=container_ready_poll_s,
    )


class TestStart:
    @pytest.mark.asyncio
    async def test_start_then_stop_roundtrips(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            assert cfg.adapter.is_running("thorn-agent-agent-x")
            assert cfg.host_control_dir.is_dir()
        finally:
            await host.stop()
        assert not cfg.adapter.is_running("thorn-agent-agent-x")

    @pytest.mark.asyncio
    async def test_double_start_raises(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            with pytest.raises(RuntimeError, match="already-running"):
                await host.start()
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_pre_existing_container_is_removed(
        self, tmp_path: Path,
    ) -> None:
        # Simulate the gateway-restart-after-crash case: a container
        # under our name is left behind by the previous gateway.
        adapter = FakeOCIRuntimeAdapter(present_images=["img"])
        from thorn.sandbox import ContainerSpec
        await adapter.run(ContainerSpec(image="img", name="thorn-agent-agent-x"))
        cfg = _make_config(tmp_path, image="img", adapter=adapter)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            # Two run calls (the pre-existing container then ours), and
            # at least one explicit remove happened before the second run.
            assert len(adapter.run_calls) == 2
            assert "thorn-agent-agent-x" in adapter.remove_calls
        finally:
            await host.stop()


class TestImageMissing:
    @pytest.mark.asyncio
    async def test_missing_image_raises_with_remediation(
        self, tmp_path: Path,
    ) -> None:
        adapter = FakeOCIRuntimeAdapter()  # no images
        cfg = _make_config(tmp_path, image="ghost:1", adapter=adapter)
        host = ContainerDaemonHost(cfg)
        with pytest.raises(SandboxImageMissingError) as exc_info:
            await host.start()
        message = str(exc_info.value)
        assert "ghost:1" in message
        assert "thorn sandbox build" in message
        # No container should have been started.
        assert adapter.run_calls == []


class TestReadinessProbe:
    @pytest.mark.asyncio
    async def test_terminal_state_raises_not_ready(self, tmp_path: Path) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["img"])
        cfg = _make_config(tmp_path, image="img", adapter=adapter)
        host = ContainerDaemonHost(cfg)

        # Drop a hook into the adapter: as soon as run() returns, flip
        # the state to ``exited`` so the readiness probe sees a
        # terminal state and bails out.
        original_run = adapter.run

        async def run_then_die(spec):
            cid = await original_run(spec)
            adapter.set_state(spec.name, status="exited", running=False, exit_code=2)
            return cid

        adapter.run = run_then_die  # type: ignore[assignment]

        with pytest.raises(ContainerNotReadyError, match="exited"):
            await host.start()
        # Teardown still happens so we don't leak.
        assert "thorn-agent-agent-x" in adapter.remove_calls

    @pytest.mark.asyncio
    async def test_timeout_raises_start_timeout(self, tmp_path: Path) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["img"])
        cfg = _make_config(
            tmp_path, image="img", adapter=adapter,
            container_ready_timeout_s=0.05,
            container_ready_poll_s=0.01,
        )
        host = ContainerDaemonHost(cfg)

        # Simulate a stuck-in-created container: never running, never
        # terminal.
        original_run = adapter.run

        async def run_then_stuck(spec):
            cid = await original_run(spec)
            adapter.set_state(spec.name, status="created", running=False)
            return cid

        adapter.run = run_then_stuck  # type: ignore[assignment]

        with pytest.raises(ContainerStartTimeoutError, match="created"):
            await host.start()


class TestSpecConstruction:
    @pytest.mark.asyncio
    async def test_mounts_cover_home_workspace_control(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            targets = {str(m.target): m for m in spec.mounts}
            assert CONTAINER_HOME_DIR in targets
            assert CONTAINER_WORKSPACE_DIR in targets
            assert CONTAINER_CONTROL_DIR in targets
            assert all(not m.read_only for m in spec.mounts)
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_dev_mount_runtime_adds_readonly_mount_and_pythonpath(
        self, tmp_path: Path,
    ) -> None:
        runtime_src = tmp_path / "src"
        runtime_src.mkdir()
        cfg = _make_config(tmp_path, dev_mount_runtime=runtime_src)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            ro_mounts = [m for m in spec.mounts if m.read_only]
            assert len(ro_mounts) == 1
            assert ro_mounts[0].source == runtime_src
            assert str(ro_mounts[0].target) == CONTAINER_RUNTIME_DIR
            env_dict = dict(spec.env)
            assert env_dict.get("PYTHONPATH") == CONTAINER_RUNTIME_DIR
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_env_passthrough_skips_unset_and_includes_set(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("THORN_TEST_PASSED", "value-1")
        monkeypatch.delenv("THORN_TEST_MISSING", raising=False)

        cfg = _make_config(
            tmp_path,
            env_passthrough=("THORN_TEST_PASSED", "THORN_TEST_MISSING"),
            extra_env=(("LITERAL", "value-2"),),
        )
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            env_dict = dict(spec.env)
            assert env_dict.get("THORN_TEST_PASSED") == "value-1"
            assert "THORN_TEST_MISSING" not in env_dict
            assert env_dict.get("LITERAL") == "value-2"
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_command_includes_in_container_paths(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            joined = " ".join(spec.command)
            assert "--socket" in joined
            assert CONTAINER_SOCKET_PATH in joined
            assert "--home" in joined
            assert CONTAINER_HOME_DIR in joined
            assert "--workspace-root" in joined
            assert CONTAINER_WORKSPACE_DIR in joined
            assert "--agent-id" in joined
            assert "agent-x" in joined
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_default_user_emits_current_uid_gid_env(
        self, tmp_path: Path,
    ) -> None:
        """The container runs the root-then-drop entrypoint, so the
        gateway forwards the target uid/gid via env vars rather than
        ``--user``.  When no explicit user is configured, the
        current process's effective uid/gid is used so bind-mount
        writes land with the operator's ownership."""
        if not hasattr(os, "geteuid"):
            pytest.skip("no os.geteuid on this platform")
        cfg = _make_config(tmp_path, user=None)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            # ``--user`` must NOT be passed to the runtime; the
            # container boots as root long enough for the entrypoint
            # to install the broker CA.  The uid/gid the daemon
            # ultimately runs as travels through env.
            assert spec.user is None
            env_dict = dict(spec.env)
            assert env_dict.get("THORN_SANDBOX_UID") == str(os.geteuid())
            assert env_dict.get("THORN_SANDBOX_GID") == str(os.getegid())
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_explicit_user_emits_configured_uid_gid_env(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_config(tmp_path, user="4242:4343")
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            assert spec.user is None
            env_dict = dict(spec.env)
            assert env_dict["THORN_SANDBOX_UID"] == "4242"
            assert env_dict["THORN_SANDBOX_GID"] == "4343"
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_entrypoint_required_caps_always_added(
        self, tmp_path: Path,
    ) -> None:
        """The entrypoint trampoline's brief root stage needs a
        small set of capabilities; they are always appended to the
        operator's ``capabilities_add`` list so the CA install and
        ``setpriv`` succeed regardless of operator policy.  The
        trampoline clears the bounding set before execing the
        daemon, so the long-running daemon itself has no caps."""
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            for cap in ENTRYPOINT_REQUIRED_CAPS:
                assert cap in spec.capabilities_add, (
                    f"missing entrypoint-required cap {cap!r} in "
                    f"{spec.capabilities_add}"
                )
        finally:
            await host.stop()


class TestBrokerBinding:
    """Phase D: ``broker_*`` fields wire the per-agent OneCLI binding
    into the container environment.

    The unit-level surface here is the spec produced by
    ``_build_container_spec`` -- env entries and bind-mounts.  The
    higher-level test (gateway → runtime → host config) lives in
    :mod:`tests.sandbox.test_runtime_wiring`.
    """

    @pytest.mark.asyncio
    async def test_no_broker_fields_means_no_proxy_env_or_mount(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            env_dict = dict(spec.env)
            assert "HTTPS_PROXY" not in env_dict
            assert "https_proxy" not in env_dict
            assert "REQUESTS_CA_BUNDLE" not in env_dict
            targets = {str(m.target) for m in spec.mounts}
            assert "/etc/thorn/onecli-ca.pem" not in targets
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_broker_fields_emit_proxy_env_and_ca_mount(
        self, tmp_path: Path,
    ) -> None:
        from thorn.sandbox._container import (
            CONTAINER_BROKER_CA_PATH,
            NO_PROXY_DEFAULT,
        )

        ca_pem = tmp_path / "broker-ca.pem"
        ca_pem.write_text("-----BEGIN CERTIFICATE-----\n...\n")

        adapter = FakeOCIRuntimeAdapter(present_images=["img"])
        cfg = _make_config(tmp_path, image="img", adapter=adapter)
        cfg = ContainerHostConfig(
            agent_id=cfg.agent_id,
            container_name=cfg.container_name,
            image=cfg.image,
            adapter=cfg.adapter,
            host_home_dir=cfg.host_home_dir,
            host_workspace_dir=cfg.host_workspace_dir,
            host_control_dir=cfg.host_control_dir,
            env_passthrough=cfg.env_passthrough,
            extra_env=cfg.extra_env,
            broker_proxy_url="http://x:aoc_token@broker:8443/",
            broker_ca_host_path=ca_pem,
            broker_placeholder_env=(
                ("GITHUB_TOKEN", "thorn-broker-placeholder-1"),
                ("GITLAB_TOKEN", "thorn-broker-placeholder-2"),
            ),
            user=cfg.user,
            container_ready_timeout_s=cfg.container_ready_timeout_s,
            container_ready_poll_s=cfg.container_ready_poll_s,
        )
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            env_dict = dict(spec.env)

            for name in (
                "HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy",
            ):
                assert env_dict[name] == "http://x:aoc_token@broker:8443/"
            assert env_dict["NO_PROXY"] == NO_PROXY_DEFAULT
            assert env_dict["no_proxy"] == NO_PROXY_DEFAULT
            for ca_var in (
                "SSL_CERT_FILE",
                "REQUESTS_CA_BUNDLE",
                "NODE_EXTRA_CA_CERTS",
                "GIT_SSL_CAINFO",
            ):
                assert env_dict[ca_var] == CONTAINER_BROKER_CA_PATH

            assert env_dict["GITHUB_TOKEN"] == "thorn-broker-placeholder-1"
            assert env_dict["GITLAB_TOKEN"] == "thorn-broker-placeholder-2"

            ca_mounts = [
                m for m in spec.mounts
                if str(m.target) == CONTAINER_BROKER_CA_PATH
            ]
            assert len(ca_mounts) == 1
            assert ca_mounts[0].source == ca_pem
            assert ca_mounts[0].read_only is True
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_git_config_mount_and_env_emitted_when_set(
        self, tmp_path: Path,
    ) -> None:
        """Setting ``git_config_host_path`` alongside the broker fields
        bind-mounts the file at the fixed in-container gitconfig
        path and points ``GIT_CONFIG_GLOBAL`` at it so every
        in-sandbox ``git`` run picks up the placeholder extraHeader
        entries."""
        ca_pem = tmp_path / "broker-ca.pem"
        ca_pem.write_text("-----BEGIN CERTIFICATE-----\n...\n")
        gitcfg = tmp_path / "gitconfig"
        gitcfg.write_text(
            '[http "https://github.com/"]\n'
            "    extraHeader = Authorization: Basic placeholder\n",
        )

        adapter = FakeOCIRuntimeAdapter(present_images=["img"])
        base = _make_config(tmp_path, image="img", adapter=adapter)
        cfg = ContainerHostConfig(
            agent_id=base.agent_id,
            container_name=base.container_name,
            image=base.image,
            adapter=base.adapter,
            host_home_dir=base.host_home_dir,
            host_workspace_dir=base.host_workspace_dir,
            host_control_dir=base.host_control_dir,
            broker_proxy_url="http://x:tok@broker:8443/",
            broker_ca_host_path=ca_pem,
            git_config_host_path=gitcfg,
            user=base.user,
            container_ready_timeout_s=base.container_ready_timeout_s,
            container_ready_poll_s=base.container_ready_poll_s,
        )
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            env_dict = dict(spec.env)
            assert env_dict["GIT_CONFIG_GLOBAL"] == CONTAINER_GIT_CONFIG_PATH
            gc_mounts = [
                m for m in spec.mounts
                if str(m.target) == CONTAINER_GIT_CONFIG_PATH
            ]
            assert len(gc_mounts) == 1
            assert gc_mounts[0].source == gitcfg
            assert gc_mounts[0].read_only is True
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_git_config_without_broker_raises(
        self, tmp_path: Path,
    ) -> None:
        """Shipping a gitconfig with extraHeader placeholders but no
        broker to rewrite them would leak the literal placeholder
        value upstream.  The container spec builder refuses the
        combination rather than making it an operator footgun."""
        gitcfg = tmp_path / "gitconfig"
        gitcfg.write_text("")

        cfg = _make_config(tmp_path)
        cfg = ContainerHostConfig(
            agent_id=cfg.agent_id,
            container_name=cfg.container_name,
            image=cfg.image,
            adapter=cfg.adapter,
            host_home_dir=cfg.host_home_dir,
            host_workspace_dir=cfg.host_workspace_dir,
            host_control_dir=cfg.host_control_dir,
            git_config_host_path=gitcfg,
            user=cfg.user,
        )
        host = ContainerDaemonHost(cfg)
        with pytest.raises(ValueError, match="git_config_host_path"):
            await host.start()

    @pytest.mark.asyncio
    async def test_broker_proxy_without_ca_path_raises(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_config(tmp_path)
        cfg = ContainerHostConfig(
            agent_id=cfg.agent_id,
            container_name=cfg.container_name,
            image=cfg.image,
            adapter=cfg.adapter,
            host_home_dir=cfg.host_home_dir,
            host_workspace_dir=cfg.host_workspace_dir,
            host_control_dir=cfg.host_control_dir,
            broker_proxy_url="http://x:tok@broker:8443/",
            broker_ca_host_path=None,
            user=cfg.user,
        )
        host = ContainerDaemonHost(cfg)
        with pytest.raises(ValueError, match="broker_ca_host_path"):
            await host.start()


class TestEgressNetwork:
    """Phase D: ``egress_network`` is surfaced as ``--network <name>``
    in the OCI run args.  Combined with an operator-created
    ``internal: true`` network containing only the broker, this
    achieves broker-only egress without Thorn touching iptables."""

    @pytest.mark.asyncio
    async def test_no_egress_network_means_no_network_flag(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            assert "--network" not in spec.extra_run_args
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_egress_network_prepends_network_flag(
        self, tmp_path: Path,
    ) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        base = _make_config(tmp_path, image="t:1", adapter=adapter)
        cfg = ContainerHostConfig(
            agent_id=base.agent_id,
            container_name=base.container_name,
            image=base.image,
            adapter=base.adapter,
            host_home_dir=base.host_home_dir,
            host_workspace_dir=base.host_workspace_dir,
            host_control_dir=base.host_control_dir,
            egress_network="thorn-broker",
            extra_run_args=("--userns=keep-id",),
            user=base.user,
            container_ready_timeout_s=base.container_ready_timeout_s,
            container_ready_poll_s=base.container_ready_poll_s,
        )
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            assert spec.extra_run_args == (
                "--network", "thorn-broker", "--userns=keep-id",
            )
        finally:
            await host.stop()


class TestSocketPath:
    def test_socket_path_under_control_dir(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        assert host.socket_path == cfg.host_control_dir / "toolhost.sock"


class TestPhaseEHardeningPassthrough:
    """Phase E: ``ContainerHostConfig`` carries hardening fields (caps,
    security_opts, read-only rootfs, tmpfs mounts, resource limits)
    and ``_build_container_spec`` threads them through to the
    :class:`ContainerSpec` verbatim.  These tests pin the
    pass-through so a future field addition can't silently drop
    something."""

    @pytest.mark.asyncio
    async def test_caps_threaded_through(self, tmp_path: Path) -> None:
        from thorn.sandbox import Tmpfs

        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        base = _make_config(tmp_path, image="t:1", adapter=adapter)
        cfg = ContainerHostConfig(
            agent_id=base.agent_id,
            container_name=base.container_name,
            image=base.image,
            adapter=base.adapter,
            host_home_dir=base.host_home_dir,
            host_workspace_dir=base.host_workspace_dir,
            host_control_dir=base.host_control_dir,
            user=base.user,
            container_ready_timeout_s=base.container_ready_timeout_s,
            container_ready_poll_s=base.container_ready_poll_s,
            capabilities_drop=("ALL",),
            capabilities_add=("NET_RAW",),
            security_opts=("no-new-privileges",),
            read_only_root=True,
            tmpfs_mounts=(
                Tmpfs(target=Path("/tmp"), options="size=1G"),
            ),
            memory_limit="2G",
            cpu_limit=2.0,
            pid_limit=512,
        )
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            assert spec.capabilities_drop == ("ALL",)
            # The operator's caps come first, then the entrypoint's
            # always-required set (de-duped).  That ordering makes
            # operator policy easy to spot in rendered run args.
            assert spec.capabilities_add[0] == "NET_RAW"
            for cap in ENTRYPOINT_REQUIRED_CAPS:
                assert cap in spec.capabilities_add
            assert spec.security_opts == ("no-new-privileges",)
            assert spec.read_only_root is True
            assert len(spec.tmpfs_mounts) == 1
            assert spec.tmpfs_mounts[0].target == Path("/tmp")
            assert spec.memory_limit == "2G"
            assert spec.cpu_limit == 2.0
            assert spec.pid_limit == 512
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_default_unset_fields_pass_through_as_unset(
        self, tmp_path: Path,
    ) -> None:
        # When a caller leaves the Phase-E fields at their dataclass
        # defaults (no hardening), the resulting ``ContainerSpec``
        # carries the same no-op defaults for everything except
        # ``capabilities_add``: the entrypoint trampoline's brief
        # root stage always needs ``SETUID``/``SETGID``/...``
        # irrespective of operator policy, so those are appended
        # unconditionally.  The trampoline clears the bounding set
        # before execing the daemon, so the hardening intent is
        # preserved from the daemon's perspective.
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            assert spec.capabilities_drop == ()
            assert spec.capabilities_add == ENTRYPOINT_REQUIRED_CAPS
            assert spec.security_opts == ()
            assert spec.read_only_root is False
            assert spec.tmpfs_mounts == ()
            assert spec.memory_limit is None
            assert spec.cpu_limit is None
            assert spec.pid_limit is None
        finally:
            await host.stop()


class TestDeriveContainerName:
    @pytest.mark.parametrize(
        "agent_id,expected",
        [
            ("simple", "thorn-agent-simple"),
            ("with.dot", "thorn-agent-with.dot"),
            ("a/b", "thorn-agent-a_b"),
            ("a%20b", "thorn-agent-a_20b"),
        ],
    )
    def test_round_trips_safe_ids(self, agent_id: str, expected: str) -> None:
        assert derive_container_name(agent_id) == expected
