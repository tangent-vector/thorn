from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_project_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _section_between(text: str, start_heading: str, end_heading: str) -> str:
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    return text[start:end]


def _single_spaced(text: str) -> str:
    return " ".join(text.split())


def test_readme_documents_handoff_based_source_acknowledgement() -> None:
    readme = _read_project_file("README.md")
    section = _section_between(
        readme,
        "### Source acknowledgement and recovery",
        "\nConfiguration\n-------------",
    )
    normalized = _single_spaced(section)

    assert "handoff-based acknowledgement contract" in normalized
    assert "has not finished the corresponding inbox work" in normalized
    assert "GitLab TODOs" in normalized
    assert "GitHub notification threads" in normalized
    assert "skips acknowledgement" in normalized
    assert "durable inbox is the source of truth for recovery" in normalized
    assert "thorn status" in normalized
    assert "thorn inbox list" in normalized
    assert "thorn inbox show" in normalized
    assert "thorn inbox requeue" in normalized
    assert "does not recreate an upstream TODO or GitHub notification" in normalized


def test_startup_flow_documents_source_specific_acknowledgement() -> None:
    startup_flow = _read_project_file("docs/startup_flow.md")
    section = _section_between(
        startup_flow,
        "## Source acknowledgement contract",
        "\n## Operator status and inbox recovery",
    )
    normalized = _single_spaced(section)

    assert "after successful handoff to the gateway" in normalized
    assert "not after the agent finishes local inbox work" in normalized
    assert "GitLab TODOs are marked done" in normalized
    assert "GitLab project-event polling is read-only" in normalized
    assert "GitHub notification threads are marked read" in normalized
    assert "thorn serve preflight" in normalized
    assert "does not read, mark, or drain GitHub/GitLab notifications" in normalized
    assert "recovery is local" in normalized
