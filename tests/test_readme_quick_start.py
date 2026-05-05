from __future__ import annotations

from pathlib import Path

README_PATH = Path(__file__).resolve().parents[1] / "README.md"


def _readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _section_between(text: str, start_heading: str, end_heading: str) -> str:
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    return text[start:end]


def test_quick_start_uses_current_host_side_gateway_flow() -> None:
    readme = _readme_text()
    quick_start = _section_between(
        readme,
        "Quick Start\n-----------",
        "\n### Advanced project configuration",
    )

    assert "uv sync --all-extras" in quick_start
    assert (
        "git clone https://gitlab-master.nvidia.com/tfoley/thorn.git"
        in quick_start
    )
    assert (
        "git clone https://github.com/tangent-vector/thorn.git"
        not in quick_start
    )
    assert "$ uv run thorn serve bootstrap \\" in quick_start
    assert "--project-url https://github.com/owner/repo" in quick_start
    assert "--agency-home ~/.thorn" in quick_start
    assert "--agency-workspace ~/thorn-workspace" in quick_start
    assert "~/.thorn/agents/my-coordinator/agent.json" in quick_start

    preflight_command = "$ uv run thorn serve --agency ~/.thorn preflight"
    serve_command = "$ uv run thorn serve --agency ~/.thorn\n"
    assert quick_start.index(preflight_command) < quick_start.index(serve_command)


def test_quick_start_does_not_reintroduce_stale_bootstrap_flags() -> None:
    readme = _readme_text()

    for stale_flag in (
        "--clone-url",
        "--native-project-id",
        "--forge-type",
        "--forge-base-url",
    ):
        assert stale_flag not in readme


def test_container_gateway_guidance_is_secondary() -> None:
    readme = _readme_text()
    quick_start = _section_between(
        readme,
        "Quick Start\n-----------",
        "\nDeployment modes\n----------------",
    )

    assert "Quick Start (Docker)" not in readme
    assert "docker run" not in quick_start
    assert "Mode B: gateway in a container alongside the broker" in readme
    assert "/var/run/docker.sock" in readme
