"""Tests for evaluation.scenario — scenario config loading and inheritance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVALUATION_DIR = Path(__file__).resolve().parent.parent / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

from scenario import (  # noqa: E402
    ScenarioConfig,
    ScenarioName,
    bootstrap_scenario,
    load_effective_prompt,
    load_scenario_config,
    overlay_template,
    resolve_inheritance_chain,
)


@pytest.fixture()
def scenarios_dir(tmp_path: Path) -> Path:
    """Return a temporary scenarios root directory."""
    d = tmp_path / "scenarios"
    d.mkdir()
    return d


def _make_scenario(
    scenarios_dir: Path,
    name: str,
    *,
    yaml_content: str | None = None,
    template_files: dict[str, str] | None = None,
) -> Path:
    """Create a scenario directory with optional yaml and template files."""
    scenario_dir = scenarios_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    if yaml_content is not None:
        (scenario_dir / "scenario.yaml").write_text(yaml_content, encoding="utf-8")

    if template_files:
        for rel_path, content in template_files.items():
            full = scenario_dir / "template" / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")

    return scenario_dir


# ---------------------------------------------------------------------------
# load_scenario_config
# ---------------------------------------------------------------------------


class TestLoadScenarioConfig:
    def test_inline_prompt(self, scenarios_dir: Path) -> None:
        d = _make_scenario(scenarios_dir, "simple", yaml_content='prompt: "do the thing"')
        cfg = load_scenario_config(d)
        assert cfg.prompt == "do the thing"
        assert cfg.prompt_file is None
        assert cfg.inherits is None

    def test_prompt_file(self, scenarios_dir: Path) -> None:
        d = _make_scenario(scenarios_dir, "with_file", yaml_content='prompt_file: "task.md"')
        (d / "task.md").write_text("  task from file  ", encoding="utf-8")
        cfg = load_scenario_config(d)
        assert cfg.prompt is None
        assert cfg.prompt_file == "task.md"

    def test_inherits_string(self, scenarios_dir: Path) -> None:
        d = _make_scenario(
            scenarios_dir, "child",
            yaml_content='prompt: "extend it"\ninherits: parent',
        )
        cfg = load_scenario_config(d)
        assert cfg.inherits == ScenarioName("parent")

    def test_inherits_single_element_list(self, scenarios_dir: Path) -> None:
        d = _make_scenario(
            scenarios_dir, "child",
            yaml_content='prompt: "extend"\ninherits:\n  - parent',
        )
        cfg = load_scenario_config(d)
        assert cfg.inherits == ScenarioName("parent")

    def test_multiline_prompt_stripped(self, scenarios_dir: Path) -> None:
        d = _make_scenario(
            scenarios_dir, "multi",
            yaml_content='prompt: |\n  hello\n  world\n',
        )
        cfg = load_scenario_config(d)
        assert cfg.prompt == "hello\nworld"

    def test_missing_yaml_raises(self, scenarios_dir: Path) -> None:
        d = scenarios_dir / "no_yaml"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="scenario.yaml"):
            load_scenario_config(d)

    def test_prompt_and_prompt_file_mutually_exclusive(
        self, scenarios_dir: Path,
    ) -> None:
        d = _make_scenario(
            scenarios_dir, "both",
            yaml_content='prompt: "inline"\nprompt_file: "task.md"',
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_scenario_config(d)

    def test_unknown_keys_rejected(self, scenarios_dir: Path) -> None:
        d = _make_scenario(
            scenarios_dir, "extra",
            yaml_content='prompt: "ok"\ngarbage: true',
        )
        with pytest.raises(ValueError, match="unknown keys"):
            load_scenario_config(d)

    def test_inherits_multi_element_list_rejected(
        self, scenarios_dir: Path,
    ) -> None:
        d = _make_scenario(
            scenarios_dir, "multi_parent",
            yaml_content='prompt: "x"\ninherits:\n  - a\n  - b',
        )
        with pytest.raises(ValueError, match="not yet supported"):
            load_scenario_config(d)

    def test_empty_yaml_is_valid(self, scenarios_dir: Path) -> None:
        d = _make_scenario(scenarios_dir, "empty", yaml_content="")
        cfg = load_scenario_config(d)
        assert cfg.prompt is None
        assert cfg.prompt_file is None
        assert cfg.inherits is None


# ---------------------------------------------------------------------------
# resolve_inheritance_chain
# ---------------------------------------------------------------------------


class TestResolveInheritanceChain:
    def test_single_scenario_no_inheritance(self, scenarios_dir: Path) -> None:
        _make_scenario(scenarios_dir, "leaf", yaml_content='prompt: "go"')
        chain = resolve_inheritance_chain(ScenarioName("leaf"), scenarios_dir)
        assert len(chain) == 1
        assert chain[0].prompt == "go"

    def test_two_level_chain(self, scenarios_dir: Path) -> None:
        _make_scenario(scenarios_dir, "base", yaml_content='prompt: "base task"')
        _make_scenario(
            scenarios_dir, "derived",
            yaml_content='prompt: "derived task"\ninherits: base',
        )
        chain = resolve_inheritance_chain(ScenarioName("derived"), scenarios_dir)
        assert len(chain) == 2
        assert chain[0].scenario_dir.name == "base"
        assert chain[1].scenario_dir.name == "derived"

    def test_three_level_chain(self, scenarios_dir: Path) -> None:
        _make_scenario(scenarios_dir, "root", yaml_content='prompt: "root"')
        _make_scenario(
            scenarios_dir, "mid",
            yaml_content='inherits: root',
        )
        _make_scenario(
            scenarios_dir, "leaf",
            yaml_content='prompt: "leaf"\ninherits: mid',
        )
        chain = resolve_inheritance_chain(ScenarioName("leaf"), scenarios_dir)
        assert [c.scenario_dir.name for c in chain] == ["root", "mid", "leaf"]

    def test_cycle_detected(self, scenarios_dir: Path) -> None:
        _make_scenario(scenarios_dir, "a", yaml_content='inherits: b')
        _make_scenario(scenarios_dir, "b", yaml_content='inherits: a')
        with pytest.raises(ValueError, match="[Cc]ycle"):
            resolve_inheritance_chain(ScenarioName("a"), scenarios_dir)

    def test_self_cycle_detected(self, scenarios_dir: Path) -> None:
        _make_scenario(scenarios_dir, "loop", yaml_content='inherits: loop')
        with pytest.raises(ValueError, match="[Cc]ycle"):
            resolve_inheritance_chain(ScenarioName("loop"), scenarios_dir)

    def test_missing_parent_raises(self, scenarios_dir: Path) -> None:
        _make_scenario(
            scenarios_dir, "orphan",
            yaml_content='inherits: nonexistent',
        )
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            resolve_inheritance_chain(ScenarioName("orphan"), scenarios_dir)


# ---------------------------------------------------------------------------
# load_effective_prompt
# ---------------------------------------------------------------------------


class TestLoadEffectivePrompt:
    def test_derived_prompt_wins(self, scenarios_dir: Path) -> None:
        _make_scenario(scenarios_dir, "base", yaml_content='prompt: "base prompt"')
        _make_scenario(
            scenarios_dir, "derived",
            yaml_content='prompt: "derived prompt"\ninherits: base',
        )
        chain = resolve_inheritance_chain(ScenarioName("derived"), scenarios_dir)
        assert load_effective_prompt(chain) == "derived prompt"

    def test_falls_through_to_base(self, scenarios_dir: Path) -> None:
        _make_scenario(scenarios_dir, "base", yaml_content='prompt: "from base"')
        _make_scenario(
            scenarios_dir, "derived",
            yaml_content='inherits: base',
        )
        chain = resolve_inheritance_chain(ScenarioName("derived"), scenarios_dir)
        assert load_effective_prompt(chain) == "from base"

    def test_prompt_file_resolution(self, scenarios_dir: Path) -> None:
        d = _make_scenario(
            scenarios_dir, "fileref",
            yaml_content='prompt_file: "task.md"',
        )
        (d / "task.md").write_text("  prompt from file  ", encoding="utf-8")
        chain = resolve_inheritance_chain(ScenarioName("fileref"), scenarios_dir)
        assert load_effective_prompt(chain) == "prompt from file"

    def test_no_prompt_anywhere_raises(self, scenarios_dir: Path) -> None:
        _make_scenario(scenarios_dir, "empty", yaml_content="")
        chain = resolve_inheritance_chain(ScenarioName("empty"), scenarios_dir)
        with pytest.raises(ValueError, match="No prompt"):
            load_effective_prompt(chain)

    def test_missing_prompt_file_raises(self, scenarios_dir: Path) -> None:
        _make_scenario(
            scenarios_dir, "bad_ref",
            yaml_content='prompt_file: "missing.md"',
        )
        chain = resolve_inheritance_chain(ScenarioName("bad_ref"), scenarios_dir)
        with pytest.raises(FileNotFoundError, match="missing.md"):
            load_effective_prompt(chain)


# ---------------------------------------------------------------------------
# overlay_template / bootstrap_scenario
# ---------------------------------------------------------------------------


class TestTemplateOverlay:
    def test_overlay_copies_files(self, tmp_path: Path) -> None:
        src = tmp_path / "template"
        src.mkdir()
        (src / "a.txt").write_text("aaa", encoding="utf-8")
        sub = src / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("bbb", encoding="utf-8")

        target = tmp_path / "workspace"
        target.mkdir()
        overlay_template(src, target)

        assert (target / "a.txt").read_text(encoding="utf-8") == "aaa"
        assert (target / "sub" / "b.txt").read_text(encoding="utf-8") == "bbb"

    def test_overlay_skips_missing_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "workspace"
        target.mkdir()
        overlay_template(tmp_path / "nonexistent", target)
        assert list(target.iterdir()) == []

    def test_bootstrap_layers_base_then_derived(
        self, scenarios_dir: Path, tmp_path: Path,
    ) -> None:
        _make_scenario(
            scenarios_dir, "base",
            yaml_content='prompt: "base"',
            template_files={
                "shared.txt": "from base",
                "base_only.txt": "base only",
            },
        )
        _make_scenario(
            scenarios_dir, "derived",
            yaml_content='prompt: "derived"\ninherits: base',
            template_files={
                "shared.txt": "from derived",
                "derived_only.txt": "derived only",
            },
        )

        chain = resolve_inheritance_chain(ScenarioName("derived"), scenarios_dir)
        target = tmp_path / "workspace"
        target.mkdir()
        bootstrap_scenario(chain, target)

        assert (target / "shared.txt").read_text(encoding="utf-8") == "from derived"
        assert (target / "base_only.txt").read_text(encoding="utf-8") == "base only"
        assert (target / "derived_only.txt").read_text(encoding="utf-8") == "derived only"

    def test_bootstrap_three_levels(
        self, scenarios_dir: Path, tmp_path: Path,
    ) -> None:
        _make_scenario(
            scenarios_dir, "root",
            yaml_content='prompt: "root"',
            template_files={"file.txt": "root"},
        )
        _make_scenario(
            scenarios_dir, "mid",
            yaml_content='inherits: root',
            template_files={"file.txt": "mid"},
        )
        _make_scenario(
            scenarios_dir, "leaf",
            yaml_content='prompt: "leaf"\ninherits: mid',
        )

        chain = resolve_inheritance_chain(ScenarioName("leaf"), scenarios_dir)
        target = tmp_path / "workspace"
        target.mkdir()
        bootstrap_scenario(chain, target)

        assert (target / "file.txt").read_text(encoding="utf-8") == "mid"
