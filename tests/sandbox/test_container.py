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
    CONTAINER_HOME_DIR,
    CONTAINER_RUNTIME_DIR,
    CONTAINER_SOCKET_PATH,
    CONTAINER_WORKSPACE_DIR,
    ContainerDaemonHost,
    ContainerHostConfig,
    ContainerNotReadyError,
    ContainerStartTimeoutError,
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
    async def test_default_user_is_current_uid_gid(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_config(tmp_path, user=None)
        host = ContainerDaemonHost(cfg)
        await host.start()
        try:
            spec = cfg.adapter.container_spec("thorn-agent-agent-x")
            if hasattr(os, "geteuid"):
                expected = f"{os.geteuid()}:{os.getegid()}"
                assert spec.user == expected
        finally:
            await host.stop()


class TestSocketPath:
    def test_socket_path_under_control_dir(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        host = ContainerDaemonHost(cfg)
        assert host.socket_path == cfg.host_control_dir / "toolhost.sock"


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
