"""Tests for the OCI runtime adapter layer.

Most coverage lives against :class:`FakeOCIRuntimeAdapter` because the
real ``podman`` / ``docker`` adapters are deliberately thin wrappers
around the CLI; the meaningful behavior is in
:class:`ContainerDaemonHost` driving them.  See
``tests/sandbox/test_container.py`` for that side, and
``tests/sandbox/test_smoke.py`` for the opt-in real-podman smoke
test.

What we do verify here:

* selection logic (``select_oci_runtime``) for explicit config,
  auto-detect, and missing-binary cases,
* the fake adapter records calls in the order expected by the host,
* ``inspect`` returns ``None`` for unknown containers,
* ``stop`` / ``remove`` are idempotent on missing containers,
* ``run`` raises :class:`OCIImageMissing` when the image is absent
  (mirrors the real runtimes' behavior so the host code can rely on
  the exception type).
"""

from __future__ import annotations

import pytest

from thorn.sandbox._runtime import (
    ContainerSpec,
    DockerAdapter,
    FakeOCIRuntimeAdapter,
    Mount,
    OCIImageMissing,
    OCIRuntimeNotFound,
    PodmanAdapter,
    select_oci_runtime,
)


class TestFakeOCIRuntime:
    @pytest.mark.asyncio
    async def test_run_records_spec_and_marks_running(self) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["thorn-sandbox:test"])
        spec = ContainerSpec(
            image="thorn-sandbox:test",
            name="agent-x",
            mounts=(
                Mount(source="/host/home", target="/agent/home"),  # type: ignore[arg-type]
            ),
            entrypoint=("python", "-m", "thorn.toolhost"),
        )
        cid = await adapter.run(spec)
        assert cid
        assert adapter.run_calls == [spec]
        state = await adapter.inspect("agent-x")
        assert state is not None
        assert state.running is True
        assert state.status == "running"

    @pytest.mark.asyncio
    async def test_run_missing_image_raises(self) -> None:
        adapter = FakeOCIRuntimeAdapter()
        spec = ContainerSpec(image="ghost:1", name="x")
        with pytest.raises(OCIImageMissing):
            await adapter.run(spec)

    @pytest.mark.asyncio
    async def test_inspect_unknown_container_returns_none(self) -> None:
        adapter = FakeOCIRuntimeAdapter()
        assert await adapter.inspect("never") is None

    @pytest.mark.asyncio
    async def test_stop_unknown_is_silent(self) -> None:
        adapter = FakeOCIRuntimeAdapter()
        await adapter.stop("never")
        assert adapter.stop_calls == ["never"]

    @pytest.mark.asyncio
    async def test_set_state_drives_inspect(self) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["x"])
        await adapter.run(ContainerSpec(image="x", name="c1"))
        adapter.set_state("c1", status="exited", running=False, exit_code=2)
        state = await adapter.inspect("c1")
        assert state is not None
        assert state.running is False
        assert state.status == "exited"
        assert state.exit_code == 2

    @pytest.mark.asyncio
    async def test_build_marks_image_present(self) -> None:
        adapter = FakeOCIRuntimeAdapter()
        from pathlib import Path
        await adapter.build(
            context=Path("/repo"),
            dockerfile=Path("/repo/Dockerfile.sandbox"),
            tag="thorn-sandbox:dev",
        )
        assert await adapter.image_exists("thorn-sandbox:dev")

    @pytest.mark.asyncio
    async def test_list_containers_filters_by_prefix(self) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["x"])
        await adapter.run(ContainerSpec(image="x", name="thorn-agent-1"))
        await adapter.run(ContainerSpec(image="x", name="other"))
        result = await adapter.list_containers(name_prefix="thorn-agent-")
        names = {state.name for state in result}
        assert names == {"thorn-agent-1"}


class TestSelectOCIRuntime:
    def test_explicit_podman_uses_factory(self) -> None:
        called = []

        def fake_podman():
            called.append("podman")
            return FakeOCIRuntimeAdapter(name="podman")

        adapter = select_oci_runtime(
            "podman",
            podman_factory=fake_podman,
            docker_factory=lambda: pytest.fail("docker should not be called"),
        )
        assert adapter.name == "podman"
        assert called == ["podman"]

    def test_explicit_docker_uses_factory(self) -> None:
        called = []

        def fake_docker():
            called.append("docker")
            return FakeOCIRuntimeAdapter(name="docker")

        adapter = select_oci_runtime(
            "docker",
            podman_factory=lambda: pytest.fail("podman should not be called"),
            docker_factory=fake_docker,
        )
        assert adapter.name == "docker"
        assert called == ["docker"]

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="podman.*docker"):
            select_oci_runtime("crio")  # type: ignore[arg-type]

    def test_auto_detect_prefers_podman(self, monkeypatch) -> None:
        from thorn.sandbox import _runtime as runtime_mod

        def fake_which(name: str) -> str | None:
            return "/usr/bin/podman" if name == "podman" else None

        monkeypatch.setattr(runtime_mod.shutil, "which", fake_which)

        called = []
        adapter = select_oci_runtime(
            None,
            podman_factory=lambda: (called.append("p"), FakeOCIRuntimeAdapter(name="podman"))[1],
            docker_factory=lambda: pytest.fail("docker should not be called"),
        )
        assert adapter.name == "podman"
        assert called == ["p"]

    def test_auto_detect_falls_back_to_docker(self, monkeypatch) -> None:
        from thorn.sandbox import _runtime as runtime_mod

        def fake_which(name: str) -> str | None:
            return "/usr/bin/docker" if name == "docker" else None

        monkeypatch.setattr(runtime_mod.shutil, "which", fake_which)

        adapter = select_oci_runtime(
            None,
            podman_factory=lambda: pytest.fail("podman should not be called"),
            docker_factory=lambda: FakeOCIRuntimeAdapter(name="docker"),
        )
        assert adapter.name == "docker"

    def test_auto_detect_neither_raises(self, monkeypatch) -> None:
        from thorn.sandbox import _runtime as runtime_mod

        monkeypatch.setattr(runtime_mod.shutil, "which", lambda _name: None)

        with pytest.raises(OCIRuntimeNotFound, match="podman"):
            select_oci_runtime(None)


class TestRealAdapterConstructorErrors:
    def test_podman_adapter_missing_binary_raises(self, monkeypatch) -> None:
        from thorn.sandbox import _runtime as runtime_mod

        monkeypatch.setattr(runtime_mod.shutil, "which", lambda _name: None)

        with pytest.raises(OCIRuntimeNotFound, match="podman"):
            PodmanAdapter()

    def test_docker_adapter_missing_binary_raises(self, monkeypatch) -> None:
        from thorn.sandbox import _runtime as runtime_mod

        monkeypatch.setattr(runtime_mod.shutil, "which", lambda _name: None)

        with pytest.raises(OCIRuntimeNotFound, match="docker"):
            DockerAdapter()
