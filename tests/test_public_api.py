"""Tests for the intended top-level ``thorn`` public surface."""

from __future__ import annotations


def test_top_level_star_exports_script_oriented_api() -> None:
    import thorn

    exported_names = set(thorn.__all__)

    assert {
        "Agent",
        "prompt",
        "skill",
        "tool",
        "run",
        "tools",
        "ThornError",
        "SkillError",
    } <= exported_names

    assert "Runtime" not in exported_names
    assert "AgentID" not in exported_names
    assert "SessionKey" not in exported_names
    assert "wrap_function" not in exported_names
    assert "discover_tools" not in exported_names
    assert "ALL_BUILTIN_TOOLS" not in exported_names
    assert "_WrappedTool" not in exported_names


def test_broader_surfaces_live_under_explicit_packages() -> None:
    import thorn
    import thorn.tools as thorn_tools
    from thorn.runtime import AgentID, Runtime, SessionKey

    assert thorn.tools is thorn_tools
    assert thorn_tools.read_file is not None
    assert Runtime is not None
    assert AgentID is not None
    assert SessionKey is not None

    assert not hasattr(thorn, "read_file")
    assert not hasattr(thorn, "Runtime")
