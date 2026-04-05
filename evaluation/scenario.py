"""Scenario configuration loading and inheritance resolution.

Each evaluation scenario directory must contain a ``scenario.yaml`` file
that specifies at minimum a prompt (inline or via file reference).
Scenarios can inherit from other scenarios via the ``inherits`` key,
forming a chain where derived scenario templates overlay base ones.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

import yaml

ScenarioName = NewType("ScenarioName", str)

_SCENARIO_YAML = "scenario.yaml"
_TEMPLATE_DIR = "template"

_KNOWN_KEYS = frozenset({"prompt", "prompt_file", "inherits"})


@dataclass(frozen=True)
class ScenarioConfig:
    """Parsed contents of a single scenario's ``scenario.yaml``."""

    scenario_dir: Path
    prompt: str | None = None
    prompt_file: str | None = None
    inherits: ScenarioName | None = None


def load_scenario_config(scenario_dir: Path) -> ScenarioConfig:
    """Read and validate ``scenario.yaml`` from *scenario_dir*."""
    yaml_path = scenario_dir / _SCENARIO_YAML
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"No {_SCENARIO_YAML} found in scenario directory {scenario_dir}"
        )

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{yaml_path}: expected a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )

    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        raise ValueError(
            f"{yaml_path}: unknown keys: {sorted(unknown)}"
        )

    prompt = raw.get("prompt")
    prompt_file = raw.get("prompt_file")
    if prompt is not None and prompt_file is not None:
        raise ValueError(
            f"{yaml_path}: 'prompt' and 'prompt_file' are mutually exclusive"
        )
    if prompt is not None:
        prompt = str(prompt).strip()

    inherits_raw = raw.get("inherits")
    inherits: ScenarioName | None = None
    if inherits_raw is not None:
        if isinstance(inherits_raw, list):
            if len(inherits_raw) != 1:
                raise ValueError(
                    f"{yaml_path}: 'inherits' as a list must have exactly "
                    f"one element (multiple inheritance is not yet supported), "
                    f"got {len(inherits_raw)}"
                )
            inherits_raw = inherits_raw[0]
        if not isinstance(inherits_raw, str):
            raise ValueError(
                f"{yaml_path}: 'inherits' must be a string (scenario name), "
                f"got {type(inherits_raw).__name__}"
            )
        inherits = ScenarioName(inherits_raw)

    return ScenarioConfig(
        scenario_dir=scenario_dir,
        prompt=prompt,
        prompt_file=prompt_file,
        inherits=inherits,
    )


def resolve_inheritance_chain(
    scenario_name: ScenarioName,
    scenarios_dir: Path,
) -> list[ScenarioConfig]:
    """Walk ``inherits`` links and return the chain ordered base-first.

    Raises on cycles or missing parent directories.
    """
    chain: list[ScenarioConfig] = []
    visited: set[ScenarioName] = set()
    current: ScenarioName | None = scenario_name

    while current is not None:
        if current in visited:
            names_so_far = " -> ".join(
                cfg.scenario_dir.name for cfg in chain
            )
            raise ValueError(
                f"Inheritance cycle detected: {names_so_far} -> {current}"
            )
        visited.add(current)

        scenario_dir = scenarios_dir / current
        if not scenario_dir.is_dir():
            available = sorted(
                p.name for p in scenarios_dir.iterdir() if p.is_dir()
            )
            raise FileNotFoundError(
                f"Inherited scenario {current!r} not found at "
                f"{scenario_dir}. Available: {available}"
            )

        config = load_scenario_config(scenario_dir)
        chain.append(config)
        current = config.inherits

    chain.reverse()
    return chain


def load_effective_prompt(chain: list[ScenarioConfig]) -> str:
    """Return the prompt from the most-derived scenario that defines one.

    Walks from derived (end of chain) to base (start), returning the
    first prompt found -- either inline or from a referenced file.
    """
    for config in reversed(chain):
        if config.prompt is not None:
            return config.prompt
        if config.prompt_file is not None:
            path = config.scenario_dir / config.prompt_file
            if not path.is_file():
                raise FileNotFoundError(
                    f"prompt_file {config.prompt_file!r} not found "
                    f"in {config.scenario_dir}"
                )
            return path.read_text(encoding="utf-8").strip()

    names = " -> ".join(cfg.scenario_dir.name for cfg in chain)
    raise ValueError(
        f"No prompt defined in scenario chain: {names}"
    )


def overlay_template(template_dir: Path, target: Path) -> None:
    """Copy all files from *template_dir* into *target*, preserving structure."""
    if not template_dir.is_dir():
        return
    for src_path in template_dir.rglob("*"):
        if src_path.is_file():
            rel = src_path.relative_to(template_dir)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)


def bootstrap_scenario(
    chain: list[ScenarioConfig],
    target: Path,
) -> None:
    """Overlay templates from each scenario in the chain, base-first."""
    for config in chain:
        overlay_template(config.scenario_dir / _TEMPLATE_DIR, target)
