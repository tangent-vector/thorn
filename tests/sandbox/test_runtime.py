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

from pathlib import Path

import pytest

from thorn.sandbox._runtime import (
    ContainerSpec,
    DockerAdapter,
    FakeOCIRuntimeAdapter,
    Mount,
    OCIImageMissing,
    OCIRuntimeNotFound,
    PodmanAdapter,
    Tmpfs,
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


class TestMountFlagEmission:
    """The ``--mount`` flag is emitted with Docker-compatible syntax.

    Docker's ``--mount`` parser only accepts ``readonly`` (a bare
    flag); it rejects ``ro`` / ``rw`` outright with
    ``invalid field 'rw' must be a key=value pair``.  Podman accepts
    both spellings, so emitting ``readonly`` for read-only mounts
    and nothing extra for read-write mounts (``rw`` is the default)
    is the one form both runtimes parse identically.  These tests
    pin that spelling so a future refactor cannot silently re-break
    docker.
    """

    @staticmethod
    def _make_docker(monkeypatch) -> DockerAdapter:
        from thorn.sandbox import _runtime as runtime_mod

        monkeypatch.setattr(
            runtime_mod.shutil, "which", lambda name: f"/usr/bin/{name}",
        )
        return DockerAdapter()

    def test_read_write_mount_omits_ro_rw_flag(self, monkeypatch) -> None:
        adapter = self._make_docker(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            mounts=(
                Mount(
                    source=Path("/host/home"),
                    target=Path("/agent/home"),
                    read_only=False,
                ),
            ),
        )
        args = list(adapter._build_run_args(spec))
        mount_arg = args[args.index("--mount") + 1]
        assert mount_arg == "type=bind,source=/host/home,target=/agent/home"
        # Docker rejects ``rw`` / ``ro`` as ``--mount`` options;
        # never emit them.
        assert ",rw" not in mount_arg
        assert ",ro" not in mount_arg

    def test_read_only_mount_uses_readonly_flag(self, monkeypatch) -> None:
        adapter = self._make_docker(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            mounts=(
                Mount(
                    source=Path("/host/ca"),
                    target=Path("/etc/ssl/ca.pem"),
                    read_only=True,
                ),
            ),
        )
        args = list(adapter._build_run_args(spec))
        mount_arg = args[args.index("--mount") + 1]
        assert mount_arg == (
            "type=bind,source=/host/ca,target=/etc/ssl/ca.pem,readonly"
        )


class TestEntrypointEmission:
    """The ``--entrypoint`` flag is emitted with Docker-compatible
    syntax: a single executable string, with any remaining argv
    elements prepended to the post-image command.  Docker's
    ``run --entrypoint`` does *not* accept the JSON-array form that
    Dockerfiles and ``podman run`` accept -- it tries to exec the
    literal JSON string as a binary path and fails with
    ``executable file not found``.  These tests pin the portable
    spelling so a future refactor cannot silently re-break docker.
    """

    @staticmethod
    def _make_docker(monkeypatch) -> DockerAdapter:
        from thorn.sandbox import _runtime as runtime_mod

        monkeypatch.setattr(
            runtime_mod.shutil, "which", lambda name: f"/usr/bin/{name}",
        )
        return DockerAdapter()

    def test_multi_arg_entrypoint_split_across_flag_and_command(
        self, monkeypatch,
    ) -> None:
        adapter = self._make_docker(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            entrypoint=("python", "-m", "thorn.toolhost"),
            command=("--socket", "/tmp/sock"),
        )
        args = list(adapter._build_run_args(spec))

        # ``--entrypoint`` carries only the executable, never a JSON
        # array string.
        ep_idx = args.index("--entrypoint")
        assert args[ep_idx + 1] == "python"
        assert "[" not in args[ep_idx + 1]

        image_idx = args.index("x:1")
        post_image = args[image_idx + 1:]
        # Remaining entrypoint args come immediately after the image,
        # then the command args.
        assert post_image == ["-m", "thorn.toolhost", "--socket", "/tmp/sock"]

    def test_single_arg_entrypoint_leaves_command_alone(
        self, monkeypatch,
    ) -> None:
        adapter = self._make_docker(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            entrypoint=("/usr/bin/myhost",),
            command=("--flag",),
        )
        args = list(adapter._build_run_args(spec))
        ep_idx = args.index("--entrypoint")
        assert args[ep_idx + 1] == "/usr/bin/myhost"
        image_idx = args.index("x:1")
        assert args[image_idx + 1:] == ["--flag"]

    def test_empty_entrypoint_tuple_rejected(self, monkeypatch) -> None:
        adapter = self._make_docker(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            entrypoint=(),
        )
        with pytest.raises(ValueError, match="non-empty"):
            list(adapter._build_run_args(spec))


class TestRuntimeLogging:
    def test_env_values_are_redacted_from_debug_command(self) -> None:
        from thorn.sandbox._runtime import _format_argv_for_log

        logged = _format_argv_for_log(
            (
                "/usr/bin/docker",
                "run",
                "-e",
                "HTTPS_PROXY=http://x:aoc_secret@onecli:10255",
                "--env=GITHUB_TOKEN=thorn-broker-placeholder",
                "image:1",
            )
        )

        assert "aoc_secret" not in logged
        assert "thorn-broker-placeholder" not in logged
        assert "HTTPS_PROXY=<redacted>" in logged
        assert "--env=GITHUB_TOKEN=<redacted>" in logged


class TestRuntimeSpecificDefaults:
    """Phase E: adapters can contribute default ``run`` args needed for
    the spec's other fields to mean what they say.  Today this only
    matters on rootless podman, where ``--userns=keep-id`` is what
    makes ``--user $(host_uid)`` actually map 1:1 to the host
    operator's uid (without it, rootless podman maps host uid X to
    in-container uid 0, so a ``--user X`` request lands on a sub-uid
    and bind-mount writes are owned by an unprivileged sub-uid the
    operator cannot read without a chown)."""

    @staticmethod
    def _make_podman(monkeypatch) -> PodmanAdapter:
        from thorn.sandbox import _runtime as runtime_mod

        monkeypatch.setattr(
            runtime_mod.shutil, "which", lambda name: f"/usr/bin/{name}",
        )
        return PodmanAdapter()

    @staticmethod
    def _make_docker(monkeypatch) -> DockerAdapter:
        from thorn.sandbox import _runtime as runtime_mod

        monkeypatch.setattr(
            runtime_mod.shutil, "which", lambda name: f"/usr/bin/{name}",
        )
        return DockerAdapter()

    def test_podman_default_keep_id_emitted(self, monkeypatch) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(image="x:1", name="agent-x", user="1000:1000")
        args = list(adapter._build_run_args(spec))
        assert "--userns=keep-id" in args

    def test_podman_default_keep_id_suppressed_when_operator_overrides(
        self, monkeypatch,
    ) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            user="1000:1000",
            extra_run_args=("--userns=auto",),
        )
        args = list(adapter._build_run_args(spec))
        assert "--userns=keep-id" not in args
        assert "--userns=auto" in args

    def test_podman_default_keep_id_suppressed_with_split_form(
        self, monkeypatch,
    ) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            user="1000:1000",
            extra_run_args=("--userns", "host"),
        )
        args = list(adapter._build_run_args(spec))
        assert "--userns=keep-id" not in args
        assert "host" in args

    def test_docker_does_not_emit_userns(self, monkeypatch) -> None:
        adapter = self._make_docker(monkeypatch)
        spec = ContainerSpec(image="x:1", name="agent-x", user="1000:1000")
        args = list(adapter._build_run_args(spec))
        assert not any(a.startswith("--userns") for a in args)


class TestPhaseEHardeningFlagEmission:
    """Phase E: ``ContainerSpec`` carries hardening fields that the
    base adapter translates to the canonical OCI run-flag spellings.
    These tests pin the flag form so a future refactor of
    :meth:`_CLIRuntimeAdapter._build_run_args` cannot silently change
    what ends up on the command line."""

    @staticmethod
    def _make_podman(monkeypatch) -> PodmanAdapter:
        from thorn.sandbox import _runtime as runtime_mod

        monkeypatch.setattr(
            runtime_mod.shutil, "which", lambda name: f"/usr/bin/{name}",
        )
        return PodmanAdapter()

    def test_capabilities_drop_emits_cap_drop(self, monkeypatch) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            capabilities_drop=("ALL",),
        )
        args = list(adapter._build_run_args(spec))
        assert "--cap-drop=ALL" in args

    def test_capabilities_add_emits_cap_add(self, monkeypatch) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            capabilities_drop=("ALL",),
            capabilities_add=("NET_RAW", "NET_BIND_SERVICE"),
        )
        args = list(adapter._build_run_args(spec))
        # cap-drop comes first, then cap-add (matches both runtimes'
        # documented evaluation order).
        drop_idx = args.index("--cap-drop=ALL")
        add_idxs = [
            args.index("--cap-add=NET_RAW"),
            args.index("--cap-add=NET_BIND_SERVICE"),
        ]
        assert all(i > drop_idx for i in add_idxs)

    def test_security_opts_emitted(self, monkeypatch) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            security_opts=("no-new-privileges", "label=disable"),
        )
        args = list(adapter._build_run_args(spec))
        assert "--security-opt=no-new-privileges" in args
        assert "--security-opt=label=disable" in args

    def test_read_only_root_flag_emitted(self, monkeypatch) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            read_only_root=True,
        )
        args = list(adapter._build_run_args(spec))
        assert "--read-only" in args

    def test_tmpfs_mounts_with_options(self, monkeypatch) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            tmpfs_mounts=(
                Tmpfs(target=Path("/tmp"), options="size=1G,mode=1777"),
                Tmpfs(target=Path("/var/tmp"), options="size=256M,mode=1777"),
            ),
        )
        args = list(adapter._build_run_args(spec))
        # Tmpfs mounts appear immediately after bind mounts, each as
        # ``--tmpfs <target>:<options>``.
        tmpfs_idxs = [i for i, a in enumerate(args) if a == "--tmpfs"]
        assert len(tmpfs_idxs) == 2
        assert args[tmpfs_idxs[0] + 1] == "/tmp:size=1G,mode=1777"
        assert args[tmpfs_idxs[1] + 1] == "/var/tmp:size=256M,mode=1777"

    def test_tmpfs_mount_without_options(self, monkeypatch) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            tmpfs_mounts=(Tmpfs(target=Path("/tmp"), options=""),),
        )
        args = list(adapter._build_run_args(spec))
        idx = args.index("--tmpfs")
        # Empty options yield ``--tmpfs <target>`` without any colon.
        assert args[idx + 1] == "/tmp"

    def test_resource_limits_emitted(self, monkeypatch) -> None:
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            memory_limit="2G",
            cpu_limit=2.0,
            pid_limit=512,
        )
        args = list(adapter._build_run_args(spec))
        memory_idx = args.index("--memory")
        assert args[memory_idx + 1] == "2G"
        cpus_idx = args.index("--cpus")
        assert args[cpus_idx + 1] == "2.0"
        pids_idx = args.index("--pids-limit")
        assert args[pids_idx + 1] == "512"

    def test_unset_hardening_fields_emit_nothing(self, monkeypatch) -> None:
        # The default ``ContainerSpec`` without Phase-E hardening
        # populated should not emit any of the corresponding flags --
        # this preserves the "test fixtures don't have to know about
        # hardening" property that lets the suite stay readable.
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(image="x:1", name="agent-x")
        args = list(adapter._build_run_args(spec))
        assert not any(a.startswith("--cap-drop") for a in args)
        assert not any(a.startswith("--cap-add") for a in args)
        assert not any(a.startswith("--security-opt") for a in args)
        assert "--read-only" not in args
        assert "--tmpfs" not in args
        assert "--memory" not in args
        assert "--cpus" not in args
        assert "--pids-limit" not in args

    def test_hardening_flags_appear_before_extra_run_args(
        self, monkeypatch,
    ) -> None:
        # The "operator escape hatch" extra_run_args should win
        # over hardening defaults via the OCI runtime's
        # last-value-wins convention -- so they must appear *after*
        # the hardening flags in the assembled argv.
        adapter = self._make_podman(monkeypatch)
        spec = ContainerSpec(
            image="x:1",
            name="agent-x",
            memory_limit="2G",
            extra_run_args=("--memory", "8G"),
        )
        args = list(adapter._build_run_args(spec))
        first_memory = args.index("--memory")
        # Find the second occurrence by searching past the first one.
        second_memory = args.index("--memory", first_memory + 1)
        assert args[first_memory + 1] == "2G"
        assert args[second_memory + 1] == "8G"
        assert second_memory > first_memory
