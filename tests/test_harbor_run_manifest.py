"""Executable checks for the versioned Harbor run-manifest contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

_HARBOR_EVAL_DIRECTORY = Path(__file__).parents[1] / "evals" / "harbor"


def _load_run_manifest_contract() -> tuple[dict[str, object], dict[str, object]]:
    schema = json.loads(
        (_HARBOR_EVAL_DIRECTORY / "run-manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    example = json.loads(
        (_HARBOR_EVAL_DIRECTORY / "run-manifest.example.json").read_text(
            encoding="utf-8"
        )
    )
    return schema, example


@pytest.mark.parametrize(
    "action_policy",
    ["baseline", "bounded-action-v1", "semantic-work-v2"],
)
def test_run_manifest_accepts_each_named_action_policy(
    action_policy: str,
) -> None:
    schema, manifest = _load_run_manifest_contract()
    agent = manifest["agent"]
    assert isinstance(agent, dict)
    agent["action_policy"] = action_policy

    Draft202012Validator(schema).validate(manifest)


def test_run_manifest_rejects_untracked_action_policy() -> None:
    schema, manifest = _load_run_manifest_contract()
    agent = manifest["agent"]
    assert isinstance(agent, dict)
    agent["action_policy"] = "untracked-experiment"

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(manifest)


@pytest.mark.parametrize(
    "validation_convergence_policy",
    [
        "baseline",
        "action-epoch-v1",
        "workspace-content-observe-v2",
        "workspace-content-v2",
    ],
)
def test_run_manifest_accepts_each_validation_convergence_policy(
    validation_convergence_policy: str,
) -> None:
    schema, manifest = _load_run_manifest_contract()
    agent = manifest["agent"]
    assert isinstance(agent, dict)
    agent["validation_convergence_policy"] = validation_convergence_policy

    Draft202012Validator(schema).validate(manifest)


def test_run_manifest_rejects_untracked_validation_convergence_policy() -> None:
    schema, manifest = _load_run_manifest_contract()
    agent = manifest["agent"]
    assert isinstance(agent, dict)
    agent["validation_convergence_policy"] = "untracked-experiment"

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(manifest)
