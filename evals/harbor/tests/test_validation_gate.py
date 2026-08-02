from __future__ import annotations

import json
import stat
import sys
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import Draft202012Validator, ValidationError


def _load_validation_gate() -> ModuleType:
    module_path = Path(__file__).parents[1] / "validation_gate.py"
    spec = spec_from_file_location("validation_gate_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load validation-gate module")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validation_gate = _load_validation_gate()
GateDisposition = validation_gate.GateDisposition
GateStopReason = validation_gate.GateStopReason
RepositoryObservation = validation_gate.RepositoryObservation
RepositoryState = validation_gate.RepositoryState
ValidationGateController = validation_gate.ValidationGateController
ValidationGateError = validation_gate.ValidationGateError
load_terminal_observation = validation_gate.load_terminal_observation
sha256 = validation_gate.sha256

REVISION = "a" * 40
MANIFEST_SCHEDULE = [
    (
        "bokeh__bokeh-13289",
        "codex-v3",
        "82a0b5e456c89563582346e8fd9e0a1e0beda4d8d6925eceab10e0a9c18c12e9",
    ),
    (
        "bokeh__bokeh-13289",
        "thorn-candidate",
        "df4e96fffd8d99ccea89edd0ee34d040f932b7b255bb76f52a541e626e5d73cb",
    ),
    (
        "conan-io__conan-11560",
        "codex-v3",
        "bb38b74d25729020c12d983e86c61715fe9f636ae357a1bd5334463d44f1b5d7",
    ),
    (
        "conan-io__conan-11560",
        "thorn-candidate",
        "c4abb0dfae205ea4c3e816da24881e6ce625423eebdf4d4ecdccbd2c449a391b",
    ),
]
SLOT_IDS = (
    "01-bokeh-codex",
    "02-bokeh-thorn",
    "03-conan-codex",
    "04-conan-thorn",
)


def _write_json(path: Path, value: object, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mode is not None:
        path.chmod(mode)


@dataclass
class _FakeRepositoryProbe:
    observation: RepositoryObservation = RepositoryObservation(
        revision=REVISION,
        state=RepositoryState.CLEAN,
    )

    def observe(self) -> RepositoryObservation:
        return self.observation


@dataclass(frozen=True)
class _FakeGate:
    controller: ValidationGateController
    definition_path: Path
    definition_sha256: str
    authorization_path: Path
    terminal_root: Path
    config_paths: tuple[Path, ...]
    task_files: tuple[Path, ...]


def _make_fake_gate(tmp_path: Path) -> _FakeGate:
    fixture_root = tmp_path / "fixture"
    declaration_path = fixture_root / "task-selection-validation-v2.md"
    declaration_path.parent.mkdir(parents=True)
    declaration_path.write_text("fake paired declaration\n", encoding="utf-8")

    task_root = fixture_root / "tasks"
    task_records: list[dict[str, object]] = []
    task_files: list[Path] = []
    task_metadata = (
        ("bokeh__bokeh-13289", "1" * 64, "a" * 64),
        ("conan-io__conan-11560", "2" * 64, "b" * 64),
    )
    task_identity_by_id = {
        task_id: (task_checksum, f"sha256:{image_digest}")
        for task_id, task_checksum, image_digest in task_metadata
    }
    for task_id, task_checksum, image_digest in task_metadata:
        package_directory = task_root / task_id
        task_file = package_directory / "task.toml"
        task_file.parent.mkdir(parents=True)
        task_file.write_text(f'name = "{task_id}"\n', encoding="utf-8")
        task_files.append(task_file)
        task_records.append(
            {
                "task_id": task_id,
                "package_directory": task_id,
                "task_checksum": task_checksum,
                "source_image": f"example/{task_id}@sha256:{image_digest}",
                "source_image_digest": f"sha256:{image_digest}",
                "files": {
                    "task.toml": {
                        "mode": "0644",
                        "sha256": sha256(task_file),
                    }
                },
            }
        )

    task_manifest_path = task_root / "manifest.json"
    _write_json(
        task_manifest_path,
        {
            "schema_version": 1,
            "schedule": {
                "entries": [
                    {
                        "order": order,
                        "task_id": task_id,
                        "arm": arm_id,
                        "selection_sha256": selection_sha256,
                    }
                    for order, (task_id, arm_id, selection_sha256) in enumerate(
                        MANIFEST_SCHEDULE,
                        start=1,
                    )
                ]
            },
            "tasks": task_records,
        },
    )

    config_paths: list[Path] = []
    schedule: list[dict[str, object]] = []
    for order, ((task_id, arm_id, selection_sha256), slot_id) in enumerate(
        zip(MANIFEST_SCHEDULE, SLOT_IDS, strict=True),
        start=1,
    ):
        config_path = fixture_root / "configs" / f"{slot_id}.json"
        _write_json(config_path, {"slot_id": slot_id, "provider": "fake"})
        config_paths.append(config_path)
        task_checksum, source_image_digest = task_identity_by_id[task_id]
        schedule.append(
            {
                "order": order,
                "block_order": 1 if order <= 2 else 2,
                "slot_id": slot_id,
                "task_id": task_id,
                "arm_id": arm_id,
                "selection_sha256": selection_sha256,
                "task_checksum": task_checksum,
                "source_image_digest": source_image_digest,
                "configuration": {
                    "path": config_path.relative_to(fixture_root).as_posix(),
                    "sha256": sha256(config_path),
                },
                "required_artifact_ids": ["job-result", "trial-result"],
            }
        )

    definition_path = fixture_root / "validation-gate-definition.json"
    definition = {
        "document_type": "validation-gate-definition",
        "schema_version": 1,
        "gate_id": "fake-validation-v2",
        "repository_revision": REVISION,
        "declaration": {
            "path": declaration_path.relative_to(fixture_root).as_posix(),
            "sha256": sha256(declaration_path),
        },
        "task_manifest": {
            "path": task_manifest_path.relative_to(fixture_root).as_posix(),
            "sha256": sha256(task_manifest_path),
        },
        "between_block_submitted_token_limit": 1_200_000,
        "schedule": schedule,
    }
    _write_json(definition_path, definition)
    definition_sha256 = sha256(definition_path)

    authorization_path = tmp_path / "authorization.json"
    _write_json(
        authorization_path,
        {
            "document_type": "validation-gate-authorization",
            "schema_version": 1,
            "gate_id": definition["gate_id"],
            "gate_definition_sha256": definition_sha256,
            "repository_revision": REVISION,
            "authorization_scope": "provider-backed-validation-v2",
            "attempt_policy": "single-no-retry",
            "authorized_slot_ids": list(SLOT_IDS),
        },
        mode=0o600,
    )

    controller = ValidationGateController.load(
        definition_path=definition_path,
        expected_definition_sha256=definition_sha256,
        state_root=tmp_path / "state",
        repository_probe=_FakeRepositoryProbe(),
    )
    return _FakeGate(
        controller=controller,
        definition_path=definition_path,
        definition_sha256=definition_sha256,
        authorization_path=authorization_path,
        terminal_root=tmp_path / "terminal",
        config_paths=tuple(config_paths),
        task_files=tuple(task_files),
    )


def _terminal_observation(
    gate: _FakeGate,
    slot_id: str,
    *,
    reward: float | None = 1.0,
    usability: str = "usable",
    terminal_outcome: str = "completed",
    submitted_tokens: int | None = 100_000,
    reported_integrity: str = "passed",
) -> Path:
    observation_directory = gate.terminal_root / slot_id
    job_result_path = observation_directory / "private-job-result.json"
    trial_result_path = observation_directory / "private-trial-result.json"
    job_result_path.parent.mkdir(parents=True, exist_ok=True)
    job_result_path.write_text(
        "raw prompt, command, and provider response must remain private\n",
        encoding="utf-8",
    )
    trial_result_path.write_text(
        "another private artifact with task content\n",
        encoding="utf-8",
    )
    observation_path = observation_directory / "terminal-observation.json"
    _write_json(
        observation_path,
        {
            "document_type": "validation-gate-terminal-observation",
            "schema_version": 1,
            "slot_id": slot_id,
            "terminal_outcome": terminal_outcome,
            "reward": reward,
            "usability": usability,
            "reported_integrity": reported_integrity,
            "submitted_tokens": submitted_tokens,
            "provider_success_count": 3,
            "harbor_retry_count": 0,
            "harness_retry_count": 0,
            "artifacts": [
                {"artifact_id": "job-result", "path": job_result_path.name},
                {"artifact_id": "trial-result", "path": trial_result_path.name},
            ],
        },
    )
    return observation_path


def _consume_and_complete(
    gate: _FakeGate,
    slot_id: str,
    **observation_overrides: object,
) -> dict[str, object]:
    gate.controller.consume_slot(
        slot_id=slot_id,
        authorization_path=gate.authorization_path,
        confirmation=f"CONSUME-{gate.controller.gate_id}-{slot_id}-NO-RETRY",
    )
    observation_path = _terminal_observation(
        gate,
        slot_id,
        **observation_overrides,
    )
    gate.controller.record_terminal(
        load_terminal_observation(observation_path),
        confirmation=f"RECORD-{gate.controller.gate_id}-{slot_id}-TERMINAL",
    )
    return json.loads(gate.controller.audit_path(slot_id).read_text(encoding="utf-8"))


def test_state_document_schema_accepts_fakes_and_rejects_raw_extra_fields(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    schema_path = Path(__file__).parents[1] / "validation-gate.schema.json"
    validator = Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    definition = json.loads(gate.definition_path.read_text(encoding="utf-8"))
    authorization = json.loads(gate.authorization_path.read_text(encoding="utf-8"))
    observation_path = _terminal_observation(gate, SLOT_IDS[0])
    observation = json.loads(observation_path.read_text(encoding="utf-8"))

    validator.validate(definition)
    validator.validate(authorization)
    validator.validate(observation)

    observation["raw_provider_response"] = "must not enter gate state"
    with pytest.raises(ValidationError):
        validator.validate(observation)


def test_validation_is_provider_inert_and_does_not_create_state(tmp_path: Path) -> None:
    gate = _make_fake_gate(tmp_path)
    gate.controller.validate_static_identities()

    assert not gate.controller.state_root.exists()
    snapshot = gate.controller.snapshot()
    assert snapshot.disposition is GateDisposition.READY
    assert snapshot.next_slot_id == SLOT_IDS[0]


def test_gate_definition_hash_and_fields_come_from_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _make_fake_gate(tmp_path)
    original_reader = validation_gate._read_file_snapshot

    def replace_after_snapshot(path: Path, label: str) -> object:
        snapshot = original_reader(path, label)
        if path == gate.definition_path:
            replacement = json.loads(path.read_text(encoding="utf-8"))
            replacement["gate_id"] = "replacement-gate"
            _write_json(path, replacement)
        return snapshot

    monkeypatch.setattr(
        validation_gate,
        "_read_file_snapshot",
        replace_after_snapshot,
    )
    controller = ValidationGateController.load(
        definition_path=gate.definition_path,
        expected_definition_sha256=gate.definition_sha256,
        state_root=tmp_path / "replacement-state",
        repository_probe=_FakeRepositoryProbe(),
    )

    assert controller.gate_id == "fake-validation-v2"
    assert sha256(gate.definition_path) != gate.definition_sha256


def test_slot_receipt_is_exclusive_immutable_and_authorization_bound(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    receipt_sha256 = gate.controller.consume_slot(
        slot_id=SLOT_IDS[0],
        authorization_path=gate.authorization_path,
        confirmation=(f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[0]}-NO-RETRY"),
    )
    receipt_path = gate.controller.receipt_path(SLOT_IDS[0])
    original = receipt_path.read_bytes()

    assert receipt_sha256 == sha256(receipt_path)
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(gate.controller.state_root.stat().st_mode) == 0o700
    schema_path = Path(__file__).parents[1] / "validation-gate.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
        json.loads(receipt_path.read_text(encoding="utf-8"))
    )
    assert gate.controller.snapshot().disposition is GateDisposition.IN_PROGRESS
    with pytest.raises(ValidationGateError, match="already consumed"):
        gate.controller.consume_slot(
            slot_id=SLOT_IDS[0],
            authorization_path=gate.authorization_path,
            confirmation=(f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[0]}-NO-RETRY"),
        )
    assert receipt_path.read_bytes() == original


def test_exclusive_receipt_write_flushes_final_mode_then_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _make_fake_gate(tmp_path)
    events: list[str] = []
    original_fchmod = validation_gate.os.fchmod
    original_fsync = validation_gate.os.fsync

    def observing_fchmod(descriptor: int, mode: int) -> None:
        events.append(f"fchmod-{mode:04o}")
        original_fchmod(descriptor, mode)

    def observing_fsync(descriptor: int) -> None:
        descriptor_mode = validation_gate.os.fstat(descriptor).st_mode
        events.append(
            "fsync-directory" if stat.S_ISDIR(descriptor_mode) else "fsync-file"
        )
        original_fsync(descriptor)

    monkeypatch.setattr(validation_gate.os, "fchmod", observing_fchmod)
    monkeypatch.setattr(validation_gate.os, "fsync", observing_fsync)

    gate.controller.consume_slot(
        slot_id=SLOT_IDS[0],
        authorization_path=gate.authorization_path,
        confirmation=f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[0]}-NO-RETRY",
    )

    assert events == ["fchmod-0400", "fsync-file", "fsync-directory"]


def test_authorization_receipt_hash_and_fields_come_from_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _make_fake_gate(tmp_path)
    original_authorization_sha256 = sha256(gate.authorization_path)
    original_loader = validation_gate._load_json_snapshot

    def replace_after_snapshot(path: Path, label: str = "JSON object") -> object:
        result = original_loader(path, label)
        if path == gate.authorization_path:
            authorization = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(
                json.dumps(authorization, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            path.chmod(0o600)
        return result

    monkeypatch.setattr(
        validation_gate,
        "_load_json_snapshot",
        replace_after_snapshot,
    )
    gate.controller.consume_slot(
        slot_id=SLOT_IDS[0],
        authorization_path=gate.authorization_path,
        confirmation=f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[0]}-NO-RETRY",
    )

    receipt = json.loads(
        gate.controller.receipt_path(SLOT_IDS[0]).read_text(encoding="utf-8")
    )
    assert receipt["authorization_receipt_sha256"] == original_authorization_sha256
    assert receipt["authorization_receipt_sha256"] != sha256(gate.authorization_path)


def test_receipt_serialization_failure_permanently_consumes_fake_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _make_fake_gate(tmp_path)

    def fail_serialization(descriptor: int, value: object) -> None:
        del descriptor, value
        raise OSError("simulated receipt serialization failure")

    monkeypatch.setattr(
        validation_gate,
        "_write_json_to_descriptor",
        fail_serialization,
    )
    with pytest.raises(OSError, match="simulated receipt serialization failure"):
        gate.controller.consume_slot(
            slot_id=SLOT_IDS[0],
            authorization_path=gate.authorization_path,
            confirmation=(f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[0]}-NO-RETRY"),
        )

    assert gate.controller.receipt_path(SLOT_IDS[0]).exists()
    with pytest.raises(ValidationGateError, match="invalid receipt"):
        gate.controller.snapshot()


def test_terminal_audit_is_content_free_exclusive_and_hash_bound(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    audit = _consume_and_complete(gate, SLOT_IDS[0])
    audit_path = gate.controller.audit_path(SLOT_IDS[0])
    original = audit_path.read_bytes()
    serialized = json.dumps(audit)

    schema_path = Path(__file__).parents[1] / "validation-gate.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
        audit
    )
    assert audit["integrity_status"] == "passed"
    assert set(audit["artifact_sha256"]) == {"job-result", "trial-result"}
    assert "raw prompt" not in serialized
    assert "private-job-result" not in serialized
    assert stat.S_IMODE(audit_path.stat().st_mode) == 0o400

    with pytest.raises(ValidationGateError, match="terminal audit already exists"):
        gate.controller.record_terminal(
            load_terminal_observation(_terminal_observation(gate, SLOT_IDS[0])),
            confirmation=(f"RECORD-{gate.controller.gate_id}-{SLOT_IDS[0]}-TERMINAL"),
        )
    assert audit_path.read_bytes() == original


def test_retained_terminal_audit_rejects_contradictory_integrity_evidence(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    audit = _consume_and_complete(gate, SLOT_IDS[0])
    audit_path = gate.controller.audit_path(SLOT_IDS[0])
    observed_identity = audit["observed_identity"]
    assert isinstance(observed_identity, dict)
    observed_identity["repository_state"] = "dirty"
    audit_path.chmod(0o600)
    _write_json(audit_path, audit, mode=0o400)

    with pytest.raises(ValidationGateError, match="invalid terminal audit"):
        gate.controller.snapshot()


def test_retained_terminal_audit_rejects_fabricated_failure(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    audit = _consume_and_complete(gate, SLOT_IDS[0])
    audit_path = gate.controller.audit_path(SLOT_IDS[0])
    audit["integrity_status"] = "failed"
    audit["integrity_failures"] = ["configuration-mismatch"]
    audit_path.chmod(0o600)
    _write_json(audit_path, audit, mode=0o400)

    with pytest.raises(ValidationGateError, match="invalid terminal audit"):
        gate.controller.snapshot()


def test_paired_blocks_complete_in_fixed_order_and_codex_reward_does_not_gate(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    _consume_and_complete(
        gate,
        SLOT_IDS[0],
        reward=0.0,
        usability="not-assessed",
        terminal_outcome="agent-exception",
        submitted_tokens=400_000,
    )
    assert gate.controller.snapshot().next_slot_id == SLOT_IDS[1]

    _consume_and_complete(
        gate,
        SLOT_IDS[1],
        reward=1.0,
        usability="usable",
        submitted_tokens=500_000,
    )
    assert gate.controller.snapshot().next_slot_id == SLOT_IDS[2]

    _consume_and_complete(
        gate,
        SLOT_IDS[2],
        reward=0.0,
        usability="not-assessed",
    )
    assert gate.controller.snapshot().next_slot_id == SLOT_IDS[3]

    _consume_and_complete(
        gate,
        SLOT_IDS[3],
        reward=0.0,
        usability="unusable",
    )
    assert gate.controller.snapshot().disposition is GateDisposition.COMPLETE


@pytest.mark.parametrize(
    ("thorn_reward", "thorn_usability", "first_tokens", "second_tokens", "reason"),
    [
        (0.0, "unusable", 100_000, 100_000, GateStopReason.THORN_BOKEH_UNUSABLE),
        (1.0, "usable", 600_001, 600_000, GateStopReason.BOKEH_TOKEN_LIMIT),
        (1.0, "usable", None, 100_000, GateStopReason.BOKEH_TOKENS_MISSING),
    ],
)
def test_between_block_gate_stops_before_conan(
    tmp_path: Path,
    thorn_reward: float,
    thorn_usability: str,
    first_tokens: int | None,
    second_tokens: int,
    reason: GateStopReason,
) -> None:
    gate = _make_fake_gate(tmp_path)
    _consume_and_complete(
        gate,
        SLOT_IDS[0],
        reward=0.0,
        usability="not-assessed",
        submitted_tokens=first_tokens,
    )
    _consume_and_complete(
        gate,
        SLOT_IDS[1],
        reward=thorn_reward,
        usability=thorn_usability,
        submitted_tokens=second_tokens,
    )

    snapshot = gate.controller.snapshot()
    assert snapshot.disposition is GateDisposition.STOPPED
    assert snapshot.stop_reason is reason
    assert snapshot.next_slot_id is None


def test_infrastructure_failure_stops_started_block_without_replacement(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    _consume_and_complete(
        gate,
        SLOT_IDS[0],
        reward=None,
        usability="not-assessed",
        terminal_outcome="infrastructure-failure",
        submitted_tokens=None,
    )

    snapshot = gate.controller.snapshot()
    assert snapshot.disposition is GateDisposition.STOPPED
    assert snapshot.stop_reason is GateStopReason.INFRASTRUCTURE_FAILURE
    with pytest.raises(ValidationGateError, match="gate is stopped"):
        gate.controller.consume_slot(
            slot_id=SLOT_IDS[1],
            authorization_path=gate.authorization_path,
            confirmation=(f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[1]}-NO-RETRY"),
        )


def test_identity_drift_blocks_consumption_and_is_terminally_audited(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    gate.task_files[0].write_text("drift before launch\n", encoding="utf-8")
    with pytest.raises(ValidationGateError, match="task-package-mismatch"):
        gate.controller.consume_slot(
            slot_id=SLOT_IDS[0],
            authorization_path=gate.authorization_path,
            confirmation=(f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[0]}-NO-RETRY"),
        )
    assert not gate.controller.receipt_path(SLOT_IDS[0]).exists()

    gate = _make_fake_gate(tmp_path / "after-consume")
    gate.controller.consume_slot(
        slot_id=SLOT_IDS[0],
        authorization_path=gate.authorization_path,
        confirmation=(f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[0]}-NO-RETRY"),
    )
    gate.config_paths[0].write_text("configuration drift\n", encoding="utf-8")
    observation = load_terminal_observation(_terminal_observation(gate, SLOT_IDS[0]))
    gate.controller.record_terminal(
        observation,
        confirmation=f"RECORD-{gate.controller.gate_id}-{SLOT_IDS[0]}-TERMINAL",
    )

    audit = json.loads(
        gate.controller.audit_path(SLOT_IDS[0]).read_text(encoding="utf-8")
    )
    assert audit["integrity_status"] == "failed"
    assert "configuration-mismatch" in audit["integrity_failures"]
    assert gate.controller.snapshot().stop_reason is GateStopReason.INTEGRITY_FAILURE


def test_task_package_root_symlink_fails_static_identity_validation(
    tmp_path: Path,
) -> None:
    gate = _make_fake_gate(tmp_path)
    package_directory = gate.task_files[0].parent
    external_directory = tmp_path / "external-package"
    package_directory.rename(external_directory)
    package_directory.symlink_to(external_directory, target_is_directory=True)

    with pytest.raises(ValidationGateError, match="task-package-mismatch"):
        gate.controller.validate_static_identities()


def test_authorization_mismatch_cannot_consume_slot(tmp_path: Path) -> None:
    gate = _make_fake_gate(tmp_path)
    authorization = json.loads(gate.authorization_path.read_text(encoding="utf-8"))
    authorization["repository_revision"] = "b" * 40
    _write_json(gate.authorization_path, authorization, mode=0o600)

    with pytest.raises(ValidationGateError, match="authorization receipt"):
        gate.controller.consume_slot(
            slot_id=SLOT_IDS[0],
            authorization_path=gate.authorization_path,
            confirmation=(f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[0]}-NO-RETRY"),
        )
    assert not gate.controller.receipt_path(SLOT_IDS[0]).exists()


def test_authorization_bytes_cannot_change_between_slots(tmp_path: Path) -> None:
    gate = _make_fake_gate(tmp_path)
    _consume_and_complete(gate, SLOT_IDS[0])
    authorization = json.loads(gate.authorization_path.read_text(encoding="utf-8"))
    gate.authorization_path.write_text(
        json.dumps(authorization, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    gate.authorization_path.chmod(0o600)

    with pytest.raises(ValidationGateError, match="bytes changed"):
        gate.controller.consume_slot(
            slot_id=SLOT_IDS[1],
            authorization_path=gate.authorization_path,
            confirmation=(f"CONSUME-{gate.controller.gate_id}-{SLOT_IDS[1]}-NO-RETRY"),
        )
    assert not gate.controller.receipt_path(SLOT_IDS[1]).exists()
