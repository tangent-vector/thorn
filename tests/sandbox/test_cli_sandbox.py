"""CLI tests for ``thorn sandbox build`` and ``thorn sandbox status``.

We monkey-patch :func:`thorn.sandbox.select_oci_runtime` to return a
:class:`FakeOCIRuntimeAdapter`, so these tests run in any
environment regardless of whether podman or docker is installed.
The "do these commands actually shell out correctly to a real
runtime?" question is the responsibility of the
``requires_podman`` / ``requires_docker`` smoke tests, not these.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from thorn._cli import main
from thorn.sandbox import FakeOCIRuntimeAdapter
from thorn.sandbox._runtime import ContainerState


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> FakeOCIRuntimeAdapter:
    adapter = FakeOCIRuntimeAdapter()

    def _select(choice: str | None = None):
        return adapter

    monkeypatch.setattr("thorn.sandbox.select_oci_runtime", _select)
    monkeypatch.setattr("thorn._cli.select_oci_runtime", _select, raising=False)
    return adapter


def _write_gateway_config(agency: Path, sandbox_block: dict | None) -> None:
    cfg: dict = {
        "providers": [{"type": "openai", "model": "gpt-4o-mini"}],
    }
    if sandbox_block is not None:
        cfg["sandbox"] = sandbox_block
    (agency / "gateway.json").write_text(json.dumps(cfg))


class TestSandboxStatus:
    def test_status_reports_missing_image(
        self,
        runner: CliRunner,
        tmp_path: Path,
        fake_adapter: FakeOCIRuntimeAdapter,
    ) -> None:
        agency = tmp_path / "agency"
        agency.mkdir()
        _write_gateway_config(agency, {"image": "thorn-sandbox:test"})

        result = runner.invoke(
            main, ["sandbox", "status", "--agency", str(agency)],
        )
        assert result.exit_code == 0, result.output
        assert "thorn-sandbox:test" in result.output
        assert "missing" in result.output
        assert "thorn sandbox build" in result.output
        assert "(none)" in result.output

    def test_status_reports_present_image(
        self,
        runner: CliRunner,
        tmp_path: Path,
        fake_adapter: FakeOCIRuntimeAdapter,
    ) -> None:
        fake_adapter._present_images.add("thorn-sandbox:test")
        agency = tmp_path / "agency"
        agency.mkdir()
        _write_gateway_config(agency, {"image": "thorn-sandbox:test"})

        result = runner.invoke(
            main, ["sandbox", "status", "--agency", str(agency)],
        )
        assert result.exit_code == 0, result.output
        assert "thorn-sandbox:test" in result.output
        assert "present" in result.output

    def test_status_lists_running_containers(
        self,
        runner: CliRunner,
        tmp_path: Path,
        fake_adapter: FakeOCIRuntimeAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _list(name_prefix: str | None = None):
            return [
                ContainerState(
                    name="thorn-agent-alpha",
                    status="running",
                    running=True,
                    exit_code=None,
                ),
                ContainerState(
                    name="thorn-agent-beta",
                    status="exited",
                    running=False,
                    exit_code=0,
                ),
            ]

        monkeypatch.setattr(fake_adapter, "list_containers", _list)
        agency = tmp_path / "agency"
        agency.mkdir()
        _write_gateway_config(agency, None)

        result = runner.invoke(
            main, ["sandbox", "status", "--agency", str(agency)],
        )
        assert result.exit_code == 0, result.output
        assert "thorn-agent-alpha" in result.output
        assert "thorn-agent-beta" in result.output
        assert "running" in result.output
        assert "exit=0" in result.output


class TestSandboxBuild:
    def test_build_invokes_adapter(
        self,
        runner: CliRunner,
        tmp_path: Path,
        fake_adapter: FakeOCIRuntimeAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Stub build_default_sandbox_image to avoid needing a real
        # Dockerfile or filesystem layout.
        async def _build(adapter, *, tag=None, dockerfile=None, context=None):
            assert adapter is fake_adapter
            return tag or "thorn-sandbox:stub"

        monkeypatch.setattr(
            "thorn.sandbox.build_default_sandbox_image", _build,
        )
        monkeypatch.setattr(
            "thorn._cli.build_default_sandbox_image", _build, raising=False,
        )

        result = runner.invoke(
            main, ["sandbox", "build", "--tag", "custom:1"],
        )
        assert result.exit_code == 0, result.output
        assert "Built sandbox image" in result.output
        assert "custom:1" in result.output
