from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

VALIDATION_V2_ROOT = Path(__file__).parents[1] / "tasks" / "validation-v2"
MANIFEST_PATH = VALIDATION_V2_ROOT / "manifest.json"
EXPECTED_TASK_IDENTITIES = {
    "bokeh__bokeh-13289": (
        "eec893cc542db1f29c2e399afedebb7baf41999eabbc310700b3ff637ec475a5",
        "sha256:ef51e22313c72834be97f6beaa3f822a"
        "ccb037cdf29832a8ab0fae1e89b0f504",
    ),
    "conan-io__conan-11560": (
        "23b3afbcee2d69bd33736a7b3fccee90685f3aa70a64633e73512ac0b1636fa3",
        "sha256:2ab97e51d7c77d280b6e818043472af5"
        "8b388043171bfb6a725deda81583b44e",
    ),
}


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declared_file_mode(path: Path) -> str:
    executable = bool(path.stat().st_mode & stat.S_IXUSR)
    return "0755" if executable else "0644"


def test_validation_v2_task_packages_match_manifest() -> None:
    manifest = _load_manifest()
    declaration_path = (VALIDATION_V2_ROOT / manifest["declaration"]).resolve()
    assert declaration_path.is_file()
    tasks = manifest["tasks"]
    declared_package_directories = {
        task["package_directory"] for task in tasks
    }
    actual_package_directories = {
        path.name
        for path in VALIDATION_V2_ROOT.iterdir()
        if path.is_dir()
    }
    assert actual_package_directories == declared_package_directories

    for task in tasks:
        task_checksum, source_image_digest = EXPECTED_TASK_IDENTITIES[
            task["task_id"]
        ]
        assert task["task_checksum"] == task_checksum
        assert task["source_image_digest"] == source_image_digest
        package_directory = VALIDATION_V2_ROOT / task["package_directory"]
        declared_files = task["files"]
        actual_files = {
            path.relative_to(package_directory).as_posix()
            for path in package_directory.rglob("*")
            if path.is_file()
        }
        assert actual_files == set(declared_files)

        for relative_path, declared_identity in declared_files.items():
            path = package_directory / relative_path
            assert _sha256(path) == declared_identity["sha256"]
            assert _declared_file_mode(path) == declared_identity["mode"]

        dockerfile = package_directory / "environment" / "Dockerfile"
        first_line = dockerfile.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == f"FROM {task['source_image']}"
        assert task["source_image"].endswith(
            f"@{task['source_image_digest']}"
        )


def test_validation_v2_schedule_hashes_are_reproducible() -> None:
    schedule = _load_manifest()["schedule"]
    seed = schedule["seed"]
    assert seed == "thorn-issue-146-heldout-paired-gate-v1"
    assert hashlib.sha256(seed.encode()).hexdigest() == schedule["seed_sha256"]

    entries = schedule["entries"]
    assert [entry["order"] for entry in entries] == [1, 2, 3, 4]
    assert [
        (entry["task_id"], entry["arm"])
        for entry in entries
    ] == [
        ("bokeh__bokeh-13289", "codex-v3"),
        ("bokeh__bokeh-13289", "thorn-candidate"),
        ("conan-io__conan-11560", "codex-v3"),
        ("conan-io__conan-11560", "thorn-candidate"),
    ]
    expected_task_block_order = [
        "bokeh__bokeh-13289",
        "conan-io__conan-11560",
    ]
    assert list(dict.fromkeys(entry["task_id"] for entry in entries)) == (
        expected_task_block_order
    )
    for task_id in expected_task_block_order:
        block_hashes = [
            entry["selection_sha256"]
            for entry in entries
            if entry["task_id"] == task_id
        ]
        assert block_hashes == sorted(block_hashes)

    for entry in entries:
        selection_key = "|".join((seed, entry["task_id"], entry["arm"]))
        assert hashlib.sha256(selection_key.encode()).hexdigest() == (
            entry["selection_sha256"]
        )


def test_validation_v2_records_provider_inert_admission() -> None:
    manifest = _load_manifest()
    canary = manifest["provider_inert_canary"]
    assert canary["completed_trial_count"] == 2
    assert canary["errored_trial_count"] == 0
    assert canary["retry_count"] == 0
    assert canary["input_tokens"] is None
    assert canary["cache_tokens"] is None
    assert canary["output_tokens"] is None
    assert canary["cost_usd"] is None
    assert canary["source_sha256"] == (
        "13ad79589828f8e4383f9b4b55e6adf8"
        "ea2fb4f926111203ce3373159b8becb3"
    )
    assert canary["config_sha256"] == (
        "73d0289bcb8420db9a2c41f23a0155e3"
        "e15ac83c0a35a653290d6671aa67a1d5"
    )
    assert canary["aggregate_result_sha256"] == (
        "26d3778a910e2e9c95cf59ba1c146a80"
        "77b9ab54e585705fdeca89e3c5100083"
    )

    assert manifest["sanitizer"] == {
        "policy": "fresh-repository-v1",
        "implementation_version": 2,
        "implementation_sha256": (
            "49fb8771e32cba650c1a2691546233564"
            "8d7b04fb79bcbf3893e0fa98b236a31"
        ),
        "workspace_metrics_collector_sha256": (
            "7f97a0400f8705a2a80060cc520a85aa"
            "ab3b0df527829432c9c819261a63719f"
        ),
    }
    for task in manifest["tasks"]:
        task_canary = task["canary"]
        assert task_canary["reward"] == 0.0
        assert task_canary["exception"] is None
        assert task_canary["source_tree"] == task_canary["result_tree"]
        assert task_canary["overlay_changed_path_count"] == 0


def test_validation_v2_harbor_patches_match_manifest() -> None:
    manifest = _load_manifest()
    assert manifest["harbor_revision"] == (
        "071281b3d931aafd6a5375fa7d5933e23054d784"
    )
    assert manifest["verifier_script_sha256"] == (
        "28ee869bc50392efc20d16705e5dd382"
        "ffd5bb6a376849d527bc8d9bbad78665"
    )

    for patch in manifest["harbor_patches"]:
        patch_path = (VALIDATION_V2_ROOT / patch["path"]).resolve()
        assert patch_path.is_file()
        assert _sha256(patch_path) == patch["sha256"]
