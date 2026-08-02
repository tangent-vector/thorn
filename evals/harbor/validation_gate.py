"""Provider-agnostic one-shot state enforcement for validation-v2.

The controller never starts Harbor or contacts a model provider. A future
arm-specific launcher must call :meth:`ValidationGateController.consume_slot`
inside its global launch lock immediately before starting Harbor, and must call
``record_terminal`` from the same broad success/failure boundary. Receipts and
audits are exclusive immutable files; this module intentionally has no reset,
delete, retry, or overwrite operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NewType, Protocol, cast

SHA256 = NewType("SHA256", str)
GitRevision = NewType("GitRevision", str)
GateID = NewType("GateID", str)
SlotID = NewType("SlotID", str)
TaskID = NewType("TaskID", str)
ArmID = NewType("ArmID", str)
ArtifactID = NewType("ArtifactID", str)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
VALIDATION_V2_TOKEN_LIMIT = 1_200_000
EXPECTED_ARM_ORDER = ("codex-v3", "thorn-candidate")


class ValidationGateError(RuntimeError):
    """A frozen gate identity or one-shot state invariant was violated."""


class RepositoryState(StrEnum):
    CLEAN = "clean"
    DIRTY = "dirty"
    UNAVAILABLE = "unavailable"


class TerminalOutcome(StrEnum):
    COMPLETED = "completed"
    AGENT_EXCEPTION = "agent-exception"
    LOOP_LIMIT = "loop-limit"
    TIMEOUT = "timeout"
    INFRASTRUCTURE_FAILURE = "infrastructure-failure"


class Usability(StrEnum):
    USABLE = "usable"
    UNUSABLE = "unusable"
    NOT_ASSESSED = "not-assessed"


class IntegrityStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class ArtifactStatus(StrEnum):
    HASHED = "hashed"
    MISSING = "missing"
    UNREADABLE = "unreadable"


class IntegrityFailure(StrEnum):
    REPORTED_INTEGRITY_FAILED = "reported-integrity-failed"
    REPOSITORY_OBSERVATION_UNAVAILABLE = "repository-observation-unavailable"
    REPOSITORY_REVISION_MISMATCH = "repository-revision-mismatch"
    REPOSITORY_WORKTREE_DIRTY = "repository-worktree-dirty"
    DECLARATION_MISMATCH = "declaration-mismatch"
    TASK_MANIFEST_MISMATCH = "task-manifest-mismatch"
    TASK_MANIFEST_CONTRACT_MISMATCH = "task-manifest-contract-mismatch"
    TASK_PACKAGE_MISMATCH = "task-package-mismatch"
    CONFIGURATION_MISMATCH = "configuration-mismatch"
    RETRY_COUNT_NONZERO = "retry-count-nonzero"
    REQUIRED_ARTIFACT_MISSING = "required-artifact-missing"
    ARTIFACT_UNREADABLE = "artifact-unreadable"
    UNEXPECTED_ARTIFACT = "unexpected-artifact"


class GateDisposition(StrEnum):
    READY = "ready"
    IN_PROGRESS = "in-progress"
    STOPPED = "stopped"
    COMPLETE = "complete"


class GateStopReason(StrEnum):
    INFRASTRUCTURE_FAILURE = "infrastructure-failure"
    INTEGRITY_FAILURE = "integrity-failure"
    THORN_BOKEH_UNUSABLE = "thorn-bokeh-unusable"
    BOKEH_TOKENS_MISSING = "bokeh-submitted-tokens-missing"
    BOKEH_TOKEN_LIMIT = "bokeh-submitted-token-limit-exceeded"


@dataclass(frozen=True)
class RepositoryObservation:
    revision: str | None
    state: RepositoryState


class RepositoryProbe(Protocol):
    def observe(self) -> RepositoryObservation:
        """Return the current revision and full tracked/untracked state."""


@dataclass(frozen=True)
class GitRepositoryProbe:
    repository_root: Path

    def observe(self) -> RepositoryObservation:
        try:
            revision = subprocess.run(
                ["git", "-C", str(self.repository_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status_output = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repository_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return RepositoryObservation(None, RepositoryState.UNAVAILABLE)
        if not GIT_REVISION_PATTERN.fullmatch(revision):
            return RepositoryObservation(None, RepositoryState.UNAVAILABLE)
        state = RepositoryState.CLEAN if not status_output else RepositoryState.DIRTY
        return RepositoryObservation(revision, state)


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: SHA256


@dataclass(frozen=True)
class FileSnapshot:
    contents: bytes
    sha256: SHA256
    mode: int


@dataclass(frozen=True)
class SlotDefinition:
    order: int
    block_order: int
    slot_id: SlotID
    task_id: TaskID
    arm_id: ArmID
    selection_sha256: SHA256
    task_checksum: SHA256
    source_image_digest: str
    configuration: FileIdentity
    required_artifact_ids: tuple[ArtifactID, ...]


@dataclass(frozen=True)
class GateDefinition:
    source_path: Path
    sha256: SHA256
    gate_id: GateID
    repository_revision: GitRevision
    declaration: FileIdentity
    task_manifest: FileIdentity
    between_block_submitted_token_limit: int
    schedule: tuple[SlotDefinition, ...]


@dataclass(frozen=True)
class TaskIdentity:
    task_id: TaskID
    task_checksum: SHA256
    source_image_digest: str


@dataclass(frozen=True)
class ArtifactInput:
    artifact_id: ArtifactID
    path: Path


@dataclass(frozen=True)
class TerminalObservation:
    slot_id: SlotID
    terminal_outcome: TerminalOutcome
    reward: float | None
    usability: Usability
    reported_integrity: IntegrityStatus
    submitted_tokens: int | None
    provider_success_count: int | None
    harbor_retry_count: int
    harness_retry_count: int
    artifacts: tuple[ArtifactInput, ...]


@dataclass(frozen=True)
class IdentityReport:
    failures: tuple[IntegrityFailure, ...]
    repository: RepositoryObservation
    declaration_sha256: SHA256 | None
    task_manifest_sha256: SHA256 | None
    configuration_sha256: SHA256 | None
    task_package_status: IntegrityStatus | RepositoryState


@dataclass(frozen=True)
class CompletedSlot:
    slot_id: SlotID
    terminal_outcome: TerminalOutcome
    reward: float | None
    usability: Usability
    integrity_status: IntegrityStatus
    submitted_tokens: int | None


@dataclass(frozen=True)
class GateSnapshot:
    disposition: GateDisposition
    completed_slot_ids: tuple[SlotID, ...]
    next_slot_id: SlotID | None = None
    active_slot_id: SlotID | None = None
    stop_reason: GateStopReason | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "completed_slot_ids": list(self.completed_slot_ids),
            "next_slot_id": self.next_slot_id,
            "active_slot_id": self.active_slot_id,
            "stop_reason": (
                self.stop_reason.value if self.stop_reason is not None else None
            ),
        }


@dataclass(frozen=True)
class ArtifactReport:
    hashes: Mapping[ArtifactID, SHA256 | None]
    statuses: Mapping[ArtifactID, ArtifactStatus]
    unexpected_count: int
    failures: frozenset[IntegrityFailure]


def sha256(path: Path) -> str:
    return _read_file_snapshot(path, "file").sha256


def _sha256_bytes(contents: bytes) -> SHA256:
    return SHA256(hashlib.sha256(contents).hexdigest())


def _read_file_snapshot(path: Path, label: str) -> FileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValidationGateError(f"{label} is unavailable: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationGateError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    except OSError as error:
        raise ValidationGateError(f"{label} is unreadable: {path}") from error
    finally:
        os.close(descriptor)
    contents = b"".join(chunks)
    return FileSnapshot(
        contents=contents,
        sha256=_sha256_bytes(contents),
        mode=stat.S_IMODE(metadata.st_mode),
    )


def _sha256_if_file(path: Path) -> SHA256 | None:
    try:
        return _read_file_snapshot(path, "file").sha256
    except ValidationGateError:
        return None


def _load_json_snapshot(
    path: Path, label: str = "JSON object"
) -> tuple[dict[str, object], FileSnapshot]:
    snapshot = _read_file_snapshot(path, label)
    try:
        value = json.loads(snapshot.contents.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValidationGateError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValidationGateError(f"invalid JSON object: {path}")
    return cast(dict[str, object], value), snapshot


def _load_json_object(path: Path) -> dict[str, object]:
    value, _ = _load_json_snapshot(path)
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected_keys: set[str],
    label: str,
) -> None:
    actual_keys = set(value)
    if actual_keys != expected_keys:
        raise ValidationGateError(
            f"{label} fields mismatch: expected {sorted(expected_keys)!r}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValidationGateError(f"{label} must be a string")
    return value


def _require_identifier(value: object, label: str) -> str:
    identifier = _require_string(value, label)
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValidationGateError(f"{label} is not a safe identifier")
    return identifier


def _require_sha256(value: object, label: str) -> SHA256:
    digest = _require_string(value, label)
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValidationGateError(f"{label} is not a SHA-256 digest")
    return SHA256(digest)


def _require_git_revision(value: object, label: str) -> GitRevision:
    revision = _require_string(value, label)
    if not GIT_REVISION_PATTERN.fullmatch(revision):
        raise ValidationGateError(f"{label} is not a full Git revision")
    return GitRevision(revision)


def _require_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationGateError(f"{label} must be an integer >= {minimum}")
    return value


def _require_optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _require_integer(value, label)


def _require_optional_reward(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationGateError("reward must be a number or null")
    reward = float(value)
    if not 0.0 <= reward <= 1.0:
        raise ValidationGateError("reward must be between zero and one")
    return reward


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValidationGateError(f"{label} must be an array")
    return cast(list[object], value)


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValidationGateError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _resolve_path(base_directory: Path, raw_path: object, label: str) -> Path:
    path_text = _require_string(raw_path, label)
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_directory / path).resolve()


def _parse_file_identity(
    value: object,
    *,
    base_directory: Path,
    label: str,
) -> FileIdentity:
    identity = _require_object(value, label)
    _require_exact_keys(identity, {"path", "sha256"}, label)
    return FileIdentity(
        path=_resolve_path(base_directory, identity["path"], f"{label} path"),
        sha256=_require_sha256(identity["sha256"], f"{label} SHA-256"),
    )


def _load_gate_definition(
    path: Path,
    expected_sha256: str,
) -> GateDefinition:
    expected_digest = _require_sha256(
        expected_sha256,
        "expected gate-definition SHA-256",
    )
    value, snapshot = _load_json_snapshot(path, "gate definition")
    if snapshot.sha256 != expected_digest:
        raise ValidationGateError("gate-definition SHA-256 mismatch")
    expected_keys = {
        "document_type",
        "schema_version",
        "gate_id",
        "repository_revision",
        "declaration",
        "task_manifest",
        "between_block_submitted_token_limit",
        "schedule",
    }
    _require_exact_keys(value, expected_keys, "gate definition")
    if value["document_type"] != "validation-gate-definition":
        raise ValidationGateError("invalid gate-definition document type")
    if value["schema_version"] != 1:
        raise ValidationGateError("unsupported gate-definition schema")
    token_limit = _require_integer(
        value["between_block_submitted_token_limit"],
        "between-block submitted-token limit",
        minimum=1,
    )
    if token_limit != VALIDATION_V2_TOKEN_LIMIT:
        raise ValidationGateError("validation-v2 token limit must be 1200000")

    schedule: list[SlotDefinition] = []
    for raw_slot in _require_list(value["schedule"], "schedule"):
        slot = _require_object(raw_slot, "schedule entry")
        _require_exact_keys(
            slot,
            {
                "order",
                "block_order",
                "slot_id",
                "task_id",
                "arm_id",
                "selection_sha256",
                "task_checksum",
                "source_image_digest",
                "configuration",
                "required_artifact_ids",
            },
            "schedule entry",
        )
        required_artifact_ids = tuple(
            ArtifactID(_require_identifier(item, "required artifact ID"))
            for item in _require_list(
                slot["required_artifact_ids"],
                "required artifact IDs",
            )
        )
        if not required_artifact_ids or len(set(required_artifact_ids)) != len(
            required_artifact_ids
        ):
            raise ValidationGateError(
                "required artifact IDs must be nonempty and unique"
            )
        schedule.append(
            SlotDefinition(
                order=_require_integer(slot["order"], "slot order", minimum=1),
                block_order=_require_integer(
                    slot["block_order"],
                    "block order",
                    minimum=1,
                ),
                slot_id=SlotID(_require_identifier(slot["slot_id"], "slot ID")),
                task_id=TaskID(_require_identifier(slot["task_id"], "task ID")),
                arm_id=ArmID(_require_identifier(slot["arm_id"], "arm ID")),
                selection_sha256=_require_sha256(
                    slot["selection_sha256"],
                    "selection SHA-256",
                ),
                task_checksum=_require_sha256(
                    slot["task_checksum"],
                    "task checksum",
                ),
                source_image_digest=_require_string(
                    slot["source_image_digest"],
                    "source image digest",
                ),
                configuration=_parse_file_identity(
                    slot["configuration"],
                    base_directory=path.parent,
                    label="slot configuration",
                ),
                required_artifact_ids=required_artifact_ids,
            )
        )

    for slot in schedule:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", slot.source_image_digest):
            raise ValidationGateError("invalid schedule source image digest")
    _validate_schedule_shape(schedule)
    return GateDefinition(
        source_path=path.resolve(),
        sha256=expected_digest,
        gate_id=GateID(_require_identifier(value["gate_id"], "gate ID")),
        repository_revision=_require_git_revision(
            value["repository_revision"],
            "repository revision",
        ),
        declaration=_parse_file_identity(
            value["declaration"],
            base_directory=path.parent,
            label="declaration",
        ),
        task_manifest=_parse_file_identity(
            value["task_manifest"],
            base_directory=path.parent,
            label="task manifest",
        ),
        between_block_submitted_token_limit=token_limit,
        schedule=tuple(schedule),
    )


def _validate_schedule_shape(schedule: Sequence[SlotDefinition]) -> None:
    if len(schedule) != 4:
        raise ValidationGateError("validation-v2 schedule must contain four slots")
    if [slot.order for slot in schedule] != [1, 2, 3, 4]:
        raise ValidationGateError("schedule orders must be exactly 1 through 4")
    if [slot.block_order for slot in schedule] != [1, 1, 2, 2]:
        raise ValidationGateError("schedule block orders must be 1, 1, 2, 2")
    if len({slot.slot_id for slot in schedule}) != len(schedule):
        raise ValidationGateError("schedule slot IDs must be unique")
    if schedule[0].task_id != schedule[1].task_id:
        raise ValidationGateError("first paired block must use one task")
    if schedule[2].task_id != schedule[3].task_id:
        raise ValidationGateError("second paired block must use one task")
    if schedule[0].task_id == schedule[2].task_id:
        raise ValidationGateError("validation-v2 task blocks must be distinct")
    for block_start in (0, 2):
        arm_order = tuple(
            slot.arm_id for slot in schedule[block_start : block_start + 2]
        )
        if arm_order != EXPECTED_ARM_ORDER:
            raise ValidationGateError(
                "each task block must order codex-v3 before thorn-candidate"
            )
        selection_hashes = tuple(
            slot.selection_sha256 for slot in schedule[block_start : block_start + 2]
        )
        if selection_hashes != tuple(sorted(selection_hashes)):
            raise ValidationGateError(
                "arm selection hashes must ascend within each task block"
            )


def load_terminal_observation(path: Path) -> TerminalObservation:
    value = _load_json_object(path)
    _require_exact_keys(
        value,
        {
            "document_type",
            "schema_version",
            "slot_id",
            "terminal_outcome",
            "reward",
            "usability",
            "reported_integrity",
            "submitted_tokens",
            "provider_success_count",
            "harbor_retry_count",
            "harness_retry_count",
            "artifacts",
        },
        "terminal observation",
    )
    if value["document_type"] != "validation-gate-terminal-observation":
        raise ValidationGateError("invalid terminal-observation document type")
    if value["schema_version"] != 1:
        raise ValidationGateError("unsupported terminal-observation schema")

    artifacts: list[ArtifactInput] = []
    for raw_artifact in _require_list(value["artifacts"], "artifacts"):
        artifact = _require_object(raw_artifact, "artifact")
        _require_exact_keys(artifact, {"artifact_id", "path"}, "artifact")
        artifacts.append(
            ArtifactInput(
                artifact_id=ArtifactID(
                    _require_identifier(artifact["artifact_id"], "artifact ID")
                ),
                path=_resolve_path(path.parent, artifact["path"], "artifact path"),
            )
        )
    if len({artifact.artifact_id for artifact in artifacts}) != len(artifacts):
        raise ValidationGateError("terminal artifact IDs must be unique")

    try:
        terminal_outcome = TerminalOutcome(
            _require_string(value["terminal_outcome"], "terminal outcome")
        )
        usability = Usability(_require_string(value["usability"], "usability"))
        reported_integrity = IntegrityStatus(
            _require_string(value["reported_integrity"], "reported integrity")
        )
    except ValueError as error:
        raise ValidationGateError("invalid terminal classification") from error
    return TerminalObservation(
        slot_id=SlotID(_require_identifier(value["slot_id"], "slot ID")),
        terminal_outcome=terminal_outcome,
        reward=_require_optional_reward(value["reward"]),
        usability=usability,
        reported_integrity=reported_integrity,
        submitted_tokens=_require_optional_integer(
            value["submitted_tokens"],
            "submitted tokens",
        ),
        provider_success_count=_require_optional_integer(
            value["provider_success_count"],
            "provider success count",
        ),
        harbor_retry_count=_require_integer(
            value["harbor_retry_count"],
            "Harbor retry count",
        ),
        harness_retry_count=_require_integer(
            value["harness_retry_count"],
            "harness retry count",
        ),
        artifacts=tuple(artifacts),
    )


def _write_json_to_descriptor(descriptor: int, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    offset = 0
    while offset < len(encoded):
        offset += os.write(descriptor, encoded[offset:])


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise ValidationGateError(f"state directory must have mode 0700: {path}")


def _write_exclusive_json(path: Path, value: object) -> None:
    _ensure_private_directory(path.parent)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValidationGateError(
            f"refusing to overwrite state file: {path}"
        ) from error
    try:
        _write_json_to_descriptor(descriptor, value)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    )
    directory_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


class _ManifestContractError(RuntimeError):
    pass


class _TaskPackageError(RuntimeError):
    pass


class ValidationGateController:
    """Derive a four-slot gate solely from immutable receipts and audits."""

    def __init__(
        self,
        *,
        definition: GateDefinition,
        state_root: Path,
        repository_probe: RepositoryProbe,
    ) -> None:
        self.definition = definition
        self.state_root = state_root
        self.repository_probe = repository_probe

    @classmethod
    def load(
        cls,
        *,
        definition_path: Path,
        expected_definition_sha256: str,
        state_root: Path,
        repository_probe: RepositoryProbe,
    ) -> ValidationGateController:
        return cls(
            definition=_load_gate_definition(
                definition_path,
                expected_definition_sha256,
            ),
            state_root=state_root,
            repository_probe=repository_probe,
        )

    @property
    def gate_id(self) -> str:
        return self.definition.gate_id

    def _slot(self, slot_id: str) -> SlotDefinition:
        for slot in self.definition.schedule:
            if slot.slot_id == slot_id:
                return slot
        raise ValidationGateError(f"unknown slot ID: {slot_id}")

    def receipt_path(self, slot_id: str) -> Path:
        slot = self._slot(slot_id)
        return self.state_root / "receipts" / f"{slot.order:02d}-{slot.slot_id}.json"

    def audit_path(self, slot_id: str) -> Path:
        slot = self._slot(slot_id)
        return self.state_root / "audits" / f"{slot.order:02d}-{slot.slot_id}.json"

    def _manifest_contract(
        self,
        manifest: Mapping[str, object],
    ) -> dict[TaskID, TaskIdentity]:
        try:
            schedule_value = _require_object(manifest["schedule"], "manifest schedule")
            raw_entries = _require_list(schedule_value["entries"], "manifest entries")
            raw_tasks = _require_list(manifest["tasks"], "manifest tasks")
        except (KeyError, ValidationGateError) as error:
            raise _ManifestContractError from error

        expected_schedule = [
            (
                slot.order,
                slot.task_id,
                slot.arm_id,
                slot.selection_sha256,
            )
            for slot in self.definition.schedule
        ]
        observed_schedule: list[tuple[int, str, str, SHA256]] = []
        try:
            for raw_entry in raw_entries:
                entry = _require_object(raw_entry, "manifest schedule entry")
                observed_schedule.append(
                    (
                        _require_integer(entry["order"], "manifest order", minimum=1),
                        _require_identifier(entry["task_id"], "manifest task ID"),
                        _require_identifier(entry["arm"], "manifest arm ID"),
                        _require_sha256(
                            entry["selection_sha256"],
                            "manifest selection SHA-256",
                        ),
                    )
                )
        except (KeyError, ValidationGateError) as error:
            raise _ManifestContractError from error
        if observed_schedule != expected_schedule:
            raise _ManifestContractError

        task_identities: dict[TaskID, TaskIdentity] = {}
        scheduled_task_ids = {slot.task_id for slot in self.definition.schedule}
        for raw_task in raw_tasks:
            task = _require_object(raw_task, "manifest task")
            try:
                task_id = TaskID(
                    _require_identifier(task["task_id"], "manifest task ID")
                )
            except (KeyError, ValidationGateError) as error:
                raise _ManifestContractError from error
            if task_id not in scheduled_task_ids or task_id in task_identities:
                raise _ManifestContractError
            try:
                task_checksum = _require_sha256(
                    task["task_checksum"],
                    "manifest task checksum",
                )
                source_image = _require_string(
                    task["source_image"],
                    "manifest source image",
                )
                source_image_digest = _require_string(
                    task["source_image_digest"],
                    "manifest source image digest",
                )
            except (KeyError, ValidationGateError) as error:
                raise _ManifestContractError from error
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", source_image_digest):
                raise _ManifestContractError
            if not source_image.endswith(f"@{source_image_digest}"):
                raise _ManifestContractError
            task_identities[task_id] = TaskIdentity(
                task_id=task_id,
                task_checksum=task_checksum,
                source_image_digest=source_image_digest,
            )
            self._verify_task_package(task)
        if set(task_identities) != scheduled_task_ids:
            raise _ManifestContractError
        for slot in self.definition.schedule:
            task_identity = task_identities[slot.task_id]
            if task_identity.task_checksum != slot.task_checksum:
                raise _ManifestContractError
            if task_identity.source_image_digest != slot.source_image_digest:
                raise _ManifestContractError
        return task_identities

    def _verify_task_package(self, task: Mapping[str, object]) -> None:
        try:
            package_directory_text = _require_string(
                task["package_directory"],
                "package directory",
            )
            declared_files = _require_object(task["files"], "task files")
        except (KeyError, ValidationGateError) as error:
            raise _ManifestContractError from error
        relative_package = Path(package_directory_text)
        if relative_package.is_absolute() or ".." in relative_package.parts:
            raise _ManifestContractError
        package_directory = self.definition.task_manifest.path.parent / relative_package
        try:
            current_directory = self.definition.task_manifest.path.parent
            for path_component in relative_package.parts:
                current_directory /= path_component
                if not stat.S_ISDIR(current_directory.lstat().st_mode):
                    raise _TaskPackageError
            if any(path.is_symlink() for path in package_directory.rglob("*")):
                raise _TaskPackageError
            actual_files = {
                path.relative_to(package_directory).as_posix()
                for path in package_directory.rglob("*")
                if path.is_file()
            }
        except OSError as error:
            raise _TaskPackageError from error
        if actual_files != set(declared_files):
            raise _TaskPackageError
        for relative_path_text, raw_identity in declared_files.items():
            relative_path = Path(relative_path_text)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise _ManifestContractError
            identity = _require_object(raw_identity, "task file identity")
            try:
                expected_hash = _require_sha256(
                    identity["sha256"],
                    "task file SHA-256",
                )
                expected_mode = _require_string(identity["mode"], "task file mode")
            except (KeyError, ValidationGateError) as error:
                raise _ManifestContractError from error
            path = package_directory / relative_path
            try:
                snapshot = _read_file_snapshot(path, "task package file")
            except ValidationGateError as error:
                raise _TaskPackageError from error
            if snapshot.sha256 != expected_hash:
                raise _TaskPackageError
            observed_mode = f"{snapshot.mode:04o}"
            if observed_mode != expected_mode:
                raise _TaskPackageError

    def _identity_report(self, slot: SlotDefinition | None = None) -> IdentityReport:
        failures: set[IntegrityFailure] = set()
        repository = self.repository_probe.observe()
        if repository.state is RepositoryState.UNAVAILABLE:
            failures.add(IntegrityFailure.REPOSITORY_OBSERVATION_UNAVAILABLE)
        else:
            if repository.revision != self.definition.repository_revision:
                failures.add(IntegrityFailure.REPOSITORY_REVISION_MISMATCH)
            if repository.state is RepositoryState.DIRTY:
                failures.add(IntegrityFailure.REPOSITORY_WORKTREE_DIRTY)

        declaration_sha256 = _sha256_if_file(self.definition.declaration.path)
        if declaration_sha256 != self.definition.declaration.sha256:
            failures.add(IntegrityFailure.DECLARATION_MISMATCH)
        try:
            manifest, manifest_snapshot = _load_json_snapshot(
                self.definition.task_manifest.path,
                "task manifest",
            )
        except ValidationGateError:
            manifest = None
            task_manifest_sha256 = None
        else:
            task_manifest_sha256 = manifest_snapshot.sha256
        task_package_status: IntegrityStatus | RepositoryState
        if task_manifest_sha256 != self.definition.task_manifest.sha256:
            failures.add(IntegrityFailure.TASK_MANIFEST_MISMATCH)
            task_package_status = RepositoryState.UNAVAILABLE
        else:
            assert manifest is not None
            try:
                self._manifest_contract(manifest)
            except _TaskPackageError:
                failures.add(IntegrityFailure.TASK_PACKAGE_MISMATCH)
                task_package_status = IntegrityStatus.FAILED
            except (ValidationGateError, _ManifestContractError):
                failures.add(IntegrityFailure.TASK_MANIFEST_CONTRACT_MISMATCH)
                task_package_status = RepositoryState.UNAVAILABLE
            else:
                task_package_status = IntegrityStatus.PASSED

        configuration_sha256 = (
            _sha256_if_file(slot.configuration.path) if slot is not None else None
        )
        for defined_slot in self.definition.schedule:
            observed_configuration = _sha256_if_file(defined_slot.configuration.path)
            if observed_configuration != defined_slot.configuration.sha256:
                failures.add(IntegrityFailure.CONFIGURATION_MISMATCH)
        return IdentityReport(
            failures=tuple(sorted(failures, key=lambda failure: failure.value)),
            repository=repository,
            declaration_sha256=declaration_sha256,
            task_manifest_sha256=task_manifest_sha256,
            configuration_sha256=configuration_sha256,
            task_package_status=task_package_status,
        )

    def validate_static_identities(self) -> None:
        report = self._identity_report()
        if report.failures:
            failures = ", ".join(failure.value for failure in report.failures)
            raise ValidationGateError(f"identity checks failed: {failures}")

    def _verify_authorization(self, path: Path) -> SHA256:
        receipt, snapshot = _load_json_snapshot(path, "authorization receipt")
        if snapshot.mode != 0o600:
            raise ValidationGateError("authorization receipt must have mode 0600")
        expected_keys = {
            "document_type",
            "schema_version",
            "gate_id",
            "gate_definition_sha256",
            "repository_revision",
            "authorization_scope",
            "attempt_policy",
            "authorized_slot_ids",
        }
        _require_exact_keys(receipt, expected_keys, "authorization receipt")
        expected_receipt: dict[str, object] = {
            "document_type": "validation-gate-authorization",
            "schema_version": 1,
            "gate_id": self.definition.gate_id,
            "gate_definition_sha256": self.definition.sha256,
            "repository_revision": self.definition.repository_revision,
            "authorization_scope": "provider-backed-validation-v2",
            "attempt_policy": "single-no-retry",
            "authorized_slot_ids": [slot.slot_id for slot in self.definition.schedule],
        }
        if receipt != expected_receipt:
            raise ValidationGateError("authorization receipt identity mismatch")
        return snapshot.sha256

    def _receipt_payload(
        self,
        slot: SlotDefinition,
        authorization_sha256: SHA256,
    ) -> dict[str, object]:
        return {
            "document_type": "validation-gate-slot-receipt",
            "schema_version": 1,
            "status": "consumed",
            "gate_id": self.definition.gate_id,
            "gate_definition_sha256": self.definition.sha256,
            "authorization_receipt_sha256": authorization_sha256,
            "repository_revision": self.definition.repository_revision,
            "declaration_sha256": self.definition.declaration.sha256,
            "task_manifest_sha256": self.definition.task_manifest.sha256,
            "slot_id": slot.slot_id,
            "order": slot.order,
            "block_order": slot.block_order,
            "task_id": slot.task_id,
            "arm_id": slot.arm_id,
            "selection_sha256": slot.selection_sha256,
            "configuration_sha256": slot.configuration.sha256,
            "task_checksum": slot.task_checksum,
            "source_image_digest": slot.source_image_digest,
        }

    def consume_slot(
        self,
        *,
        slot_id: str,
        authorization_path: Path,
        confirmation: str,
    ) -> str:
        slot = self._slot(slot_id)
        receipt_path = self.receipt_path(slot_id)
        if receipt_path.exists():
            raise ValidationGateError(f"slot already consumed: {slot_id}")
        expected_confirmation = (
            f"CONSUME-{self.definition.gate_id}-{slot.slot_id}-NO-RETRY"
        )
        if confirmation != expected_confirmation:
            raise ValidationGateError("slot consumption confirmation mismatch")
        snapshot = self.snapshot()
        if snapshot.disposition is GateDisposition.STOPPED:
            raise ValidationGateError(f"gate is stopped: {snapshot.stop_reason.value}")
        if snapshot.disposition is not GateDisposition.READY:
            raise ValidationGateError(
                f"gate is not ready for a new slot: {snapshot.disposition.value}"
            )
        if snapshot.next_slot_id != slot.slot_id:
            raise ValidationGateError(
                f"next slot is {snapshot.next_slot_id}, not {slot.slot_id}"
            )
        self.validate_static_identities()
        authorization_sha256 = self._verify_authorization(authorization_path)
        for prior_slot in self.definition.schedule[: slot.order - 1]:
            prior_receipt_path = self.receipt_path(prior_slot.slot_id)
            if not prior_receipt_path.exists():
                continue
            prior_receipt, _ = self._read_receipt(prior_slot)
            if prior_receipt["authorization_receipt_sha256"] != authorization_sha256:
                raise ValidationGateError(
                    "authorization receipt bytes changed after gate start"
                )
        _ensure_private_directory(self.state_root)
        try:
            _write_exclusive_json(
                receipt_path,
                self._receipt_payload(slot, authorization_sha256),
            )
        except ValidationGateError as error:
            if receipt_path.exists():
                raise ValidationGateError(
                    f"slot already consumed: {slot_id}"
                ) from error
            raise
        return sha256(receipt_path)

    def _read_receipt(
        self,
        slot: SlotDefinition,
    ) -> tuple[dict[str, object], SHA256]:
        path = self.receipt_path(slot.slot_id)
        try:
            receipt, snapshot = _load_json_snapshot(path, "slot receipt")
            if snapshot.mode != 0o400:
                raise ValidationGateError("slot receipt mode is invalid")
            expected_keys = set(self._receipt_payload_keys())
            _require_exact_keys(receipt, expected_keys, "slot receipt")
            if receipt["document_type"] != "validation-gate-slot-receipt":
                raise ValidationGateError("slot receipt document type is invalid")
            if receipt["schema_version"] != 1:
                raise ValidationGateError("slot receipt schema is invalid")
            if receipt["status"] != "consumed":
                raise ValidationGateError("slot receipt status is invalid")
            if receipt["gate_id"] != self.definition.gate_id:
                raise ValidationGateError("slot receipt gate mismatch")
            if receipt["gate_definition_sha256"] != self.definition.sha256:
                raise ValidationGateError("slot receipt definition mismatch")
            if receipt["repository_revision"] != self.definition.repository_revision:
                raise ValidationGateError("slot receipt revision mismatch")
            if receipt["declaration_sha256"] != self.definition.declaration.sha256:
                raise ValidationGateError("slot receipt declaration mismatch")
            if receipt["task_manifest_sha256"] != self.definition.task_manifest.sha256:
                raise ValidationGateError("slot receipt task manifest mismatch")
            if receipt["slot_id"] != slot.slot_id or receipt["order"] != slot.order:
                raise ValidationGateError("slot receipt schedule mismatch")
            if receipt["block_order"] != slot.block_order:
                raise ValidationGateError("slot receipt block mismatch")
            if receipt["task_id"] != slot.task_id or receipt["arm_id"] != slot.arm_id:
                raise ValidationGateError("slot receipt arm mismatch")
            if receipt["selection_sha256"] != slot.selection_sha256:
                raise ValidationGateError("slot receipt selection mismatch")
            if receipt["configuration_sha256"] != slot.configuration.sha256:
                raise ValidationGateError("slot receipt configuration mismatch")
            _require_sha256(
                receipt["authorization_receipt_sha256"], "authorization SHA-256"
            )
            if receipt["task_checksum"] != slot.task_checksum:
                raise ValidationGateError("slot receipt task checksum mismatch")
            source_digest = _require_string(
                receipt["source_image_digest"],
                "source image digest",
            )
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", source_digest):
                raise ValidationGateError("invalid receipt source image digest")
            if source_digest != slot.source_image_digest:
                raise ValidationGateError("slot receipt source image mismatch")
        except (KeyError, OSError, ValidationGateError) as error:
            raise ValidationGateError(
                f"invalid receipt for slot {slot.slot_id}"
            ) from error
        return receipt, snapshot.sha256

    @staticmethod
    def _receipt_payload_keys() -> tuple[str, ...]:
        return (
            "document_type",
            "schema_version",
            "status",
            "gate_id",
            "gate_definition_sha256",
            "authorization_receipt_sha256",
            "repository_revision",
            "declaration_sha256",
            "task_manifest_sha256",
            "slot_id",
            "order",
            "block_order",
            "task_id",
            "arm_id",
            "selection_sha256",
            "configuration_sha256",
            "task_checksum",
            "source_image_digest",
        )

    def _artifact_report(
        self,
        slot: SlotDefinition,
        observation: TerminalObservation,
    ) -> ArtifactReport:
        failures: set[IntegrityFailure] = set()
        supplied = {
            artifact.artifact_id: artifact.path for artifact in observation.artifacts
        }
        required = set(slot.required_artifact_ids)
        unexpected_count = len(set(supplied) - required)
        if unexpected_count:
            failures.add(IntegrityFailure.UNEXPECTED_ARTIFACT)
        hashes: dict[ArtifactID, SHA256 | None] = {}
        statuses: dict[ArtifactID, ArtifactStatus] = {}
        for artifact_id in slot.required_artifact_ids:
            path = supplied.get(artifact_id)
            if path is None:
                hashes[artifact_id] = None
                statuses[artifact_id] = ArtifactStatus.MISSING
                failures.add(IntegrityFailure.REQUIRED_ARTIFACT_MISSING)
                continue
            try:
                snapshot = _read_file_snapshot(path, "terminal artifact")
            except ValidationGateError:
                hashes[artifact_id] = None
                statuses[artifact_id] = ArtifactStatus.UNREADABLE
                failures.add(IntegrityFailure.ARTIFACT_UNREADABLE)
                continue
            hashes[artifact_id] = snapshot.sha256
            statuses[artifact_id] = ArtifactStatus.HASHED
        return ArtifactReport(
            hashes=hashes,
            statuses=statuses,
            unexpected_count=unexpected_count,
            failures=frozenset(failures),
        )

    def record_terminal(
        self,
        observation: TerminalObservation,
        *,
        confirmation: str,
    ) -> str:
        slot = self._slot(observation.slot_id)
        audit_path = self.audit_path(slot.slot_id)
        if audit_path.exists():
            raise ValidationGateError(
                f"terminal audit already exists for slot {slot.slot_id}"
            )
        expected_confirmation = (
            f"RECORD-{self.definition.gate_id}-{slot.slot_id}-TERMINAL"
        )
        if confirmation != expected_confirmation:
            raise ValidationGateError("terminal audit confirmation mismatch")
        snapshot = self.snapshot()
        if (
            snapshot.disposition is not GateDisposition.IN_PROGRESS
            or snapshot.active_slot_id != slot.slot_id
        ):
            raise ValidationGateError("slot is not the active consumed attempt")

        receipt, receipt_sha256 = self._read_receipt(slot)
        identity_report = self._identity_report(slot)
        artifact_report = self._artifact_report(
            slot,
            observation,
        )
        failures = set(identity_report.failures) | artifact_report.failures
        if observation.reported_integrity is IntegrityStatus.FAILED:
            failures.add(IntegrityFailure.REPORTED_INTEGRITY_FAILED)
        if observation.harbor_retry_count or observation.harness_retry_count:
            failures.add(IntegrityFailure.RETRY_COUNT_NONZERO)
        integrity_status = (
            IntegrityStatus.PASSED if not failures else IntegrityStatus.FAILED
        )

        audit: dict[str, object] = {
            "document_type": "validation-gate-terminal-audit",
            "schema_version": 1,
            "gate_id": self.definition.gate_id,
            "gate_definition_sha256": self.definition.sha256,
            "receipt_sha256": receipt_sha256,
            "authorization_receipt_sha256": receipt["authorization_receipt_sha256"],
            "slot_id": slot.slot_id,
            "order": slot.order,
            "block_order": slot.block_order,
            "task_id": slot.task_id,
            "arm_id": slot.arm_id,
            "selection_sha256": slot.selection_sha256,
            "terminal_outcome": observation.terminal_outcome.value,
            "reward": observation.reward,
            "usability": observation.usability.value,
            "reported_integrity": observation.reported_integrity.value,
            "integrity_status": integrity_status.value,
            "integrity_failures": [
                failure.value
                for failure in sorted(failures, key=lambda item: item.value)
            ],
            "submitted_tokens": observation.submitted_tokens,
            "provider_success_count": observation.provider_success_count,
            "harbor_retry_count": observation.harbor_retry_count,
            "harness_retry_count": observation.harness_retry_count,
            "expected_identity": {
                "repository_revision": receipt["repository_revision"],
                "declaration_sha256": receipt["declaration_sha256"],
                "task_manifest_sha256": receipt["task_manifest_sha256"],
                "configuration_sha256": receipt["configuration_sha256"],
                "task_checksum": receipt["task_checksum"],
                "source_image_digest": receipt["source_image_digest"],
            },
            "observed_identity": {
                "repository_revision": identity_report.repository.revision,
                "repository_state": identity_report.repository.state.value,
                "declaration_sha256": identity_report.declaration_sha256,
                "task_manifest_sha256": identity_report.task_manifest_sha256,
                "configuration_sha256": identity_report.configuration_sha256,
                "task_package_status": identity_report.task_package_status.value,
            },
            "artifact_sha256": artifact_report.hashes,
            "artifact_status": {
                artifact_id: status.value
                for artifact_id, status in artifact_report.statuses.items()
            },
            "unexpected_artifact_count": artifact_report.unexpected_count,
        }
        _ensure_private_directory(self.state_root)
        _write_exclusive_json(audit_path, audit)
        return sha256(audit_path)

    def _read_terminal_audit(
        self,
        slot: SlotDefinition,
        receipt: Mapping[str, object],
        receipt_sha256: SHA256,
    ) -> CompletedSlot:
        path = self.audit_path(slot.slot_id)
        try:
            audit, snapshot = _load_json_snapshot(path, "terminal audit")
            if snapshot.mode != 0o400:
                raise ValidationGateError("terminal-audit mode is invalid")
            required_keys = {
                "document_type",
                "schema_version",
                "gate_id",
                "gate_definition_sha256",
                "receipt_sha256",
                "authorization_receipt_sha256",
                "slot_id",
                "order",
                "block_order",
                "task_id",
                "arm_id",
                "selection_sha256",
                "terminal_outcome",
                "reward",
                "usability",
                "reported_integrity",
                "integrity_status",
                "integrity_failures",
                "submitted_tokens",
                "provider_success_count",
                "harbor_retry_count",
                "harness_retry_count",
                "expected_identity",
                "observed_identity",
                "artifact_sha256",
                "artifact_status",
                "unexpected_artifact_count",
            }
            _require_exact_keys(audit, required_keys, "terminal audit")
            if audit["document_type"] != "validation-gate-terminal-audit":
                raise ValidationGateError("invalid terminal-audit document type")
            if audit["schema_version"] != 1:
                raise ValidationGateError("unsupported terminal-audit schema")
            if audit["gate_id"] != self.definition.gate_id:
                raise ValidationGateError("terminal-audit gate mismatch")
            if audit["gate_definition_sha256"] != self.definition.sha256:
                raise ValidationGateError("terminal-audit definition mismatch")
            if audit["receipt_sha256"] != receipt_sha256:
                raise ValidationGateError("terminal-audit receipt mismatch")
            if (
                audit["authorization_receipt_sha256"]
                != receipt["authorization_receipt_sha256"]
            ):
                raise ValidationGateError("terminal-audit authorization mismatch")
            if (
                audit["slot_id"] != slot.slot_id
                or audit["order"] != slot.order
                or audit["block_order"] != slot.block_order
                or audit["task_id"] != slot.task_id
                or audit["arm_id"] != slot.arm_id
                or audit["selection_sha256"] != slot.selection_sha256
            ):
                raise ValidationGateError("terminal-audit schedule mismatch")
            terminal_outcome = TerminalOutcome(
                _require_string(audit["terminal_outcome"], "terminal outcome")
            )
            usability = Usability(_require_string(audit["usability"], "usability"))
            integrity_status = IntegrityStatus(
                _require_string(audit["integrity_status"], "integrity status")
            )
            reward = _require_optional_reward(audit["reward"])
            submitted_tokens = _require_optional_integer(
                audit["submitted_tokens"],
                "submitted tokens",
            )
            self._validate_terminal_audit_evidence(
                audit,
                slot=slot,
                receipt=receipt,
                integrity_status=integrity_status,
            )
        except (KeyError, OSError, ValueError, ValidationGateError) as error:
            raise ValidationGateError(
                f"invalid terminal audit for slot {slot.slot_id}"
            ) from error
        return CompletedSlot(
            slot_id=slot.slot_id,
            terminal_outcome=terminal_outcome,
            reward=reward,
            usability=usability,
            integrity_status=integrity_status,
            submitted_tokens=submitted_tokens,
        )

    def _validate_terminal_audit_evidence(
        self,
        audit: Mapping[str, object],
        *,
        slot: SlotDefinition,
        receipt: Mapping[str, object],
        integrity_status: IntegrityStatus,
    ) -> None:
        raw_failures = _require_list(
            audit["integrity_failures"],
            "integrity failures",
        )
        try:
            failures = {
                IntegrityFailure(_require_string(raw_failure, "integrity failure"))
                for raw_failure in raw_failures
            }
            reported_integrity = IntegrityStatus(
                _require_string(audit["reported_integrity"], "reported integrity")
            )
        except ValueError as error:
            raise ValidationGateError("invalid integrity failure") from error
        if len(failures) != len(raw_failures):
            raise ValidationGateError("integrity failures must be unique")

        harbor_retry_count = _require_integer(
            audit["harbor_retry_count"],
            "Harbor retry count",
        )
        harness_retry_count = _require_integer(
            audit["harness_retry_count"],
            "harness retry count",
        )
        _require_optional_integer(
            audit["provider_success_count"],
            "provider success count",
        )

        expected_identity = _require_object(
            audit["expected_identity"],
            "expected identity",
        )
        expected_identity_keys = {
            "repository_revision",
            "declaration_sha256",
            "task_manifest_sha256",
            "configuration_sha256",
            "task_checksum",
            "source_image_digest",
        }
        _require_exact_keys(
            expected_identity,
            expected_identity_keys,
            "expected identity",
        )
        receipt_identity = {key: receipt[key] for key in expected_identity_keys}
        if expected_identity != receipt_identity:
            raise ValidationGateError("terminal-audit expected identity mismatch")

        observed_identity = _require_object(
            audit["observed_identity"],
            "observed identity",
        )
        _require_exact_keys(
            observed_identity,
            {
                "repository_revision",
                "repository_state",
                "declaration_sha256",
                "task_manifest_sha256",
                "configuration_sha256",
                "task_package_status",
            },
            "observed identity",
        )
        observed_revision = observed_identity["repository_revision"]
        if observed_revision is not None:
            observed_revision = _require_git_revision(
                observed_revision,
                "observed repository revision",
            )
        try:
            repository_state = RepositoryState(
                _require_string(
                    observed_identity["repository_state"],
                    "observed repository state",
                )
            )
        except ValueError as error:
            raise ValidationGateError("invalid observed repository state") from error
        observed_digests: dict[str, SHA256 | None] = {}
        for key in (
            "declaration_sha256",
            "task_manifest_sha256",
            "configuration_sha256",
        ):
            observed_digest = observed_identity[key]
            observed_digests[key] = (
                None
                if observed_digest is None
                else _require_sha256(observed_digest, f"observed {key}")
            )
        try:
            task_package_status = IntegrityStatus(
                _require_string(
                    observed_identity["task_package_status"],
                    "task package status",
                )
            )
        except ValueError:
            if observed_identity["task_package_status"] == RepositoryState.UNAVAILABLE:
                task_package_status = RepositoryState.UNAVAILABLE
            else:
                raise ValidationGateError("invalid task package status") from None

        artifact_hashes = _require_object(
            audit["artifact_sha256"],
            "artifact SHA-256",
        )
        artifact_status_values = _require_object(
            audit["artifact_status"],
            "artifact status",
        )
        required_artifact_ids = set(slot.required_artifact_ids)
        if (
            set(artifact_hashes) != required_artifact_ids
            or set(artifact_status_values) != required_artifact_ids
        ):
            raise ValidationGateError("terminal-audit artifact IDs mismatch")
        artifact_statuses: dict[str, ArtifactStatus] = {}
        for artifact_id in slot.required_artifact_ids:
            try:
                artifact_status = ArtifactStatus(
                    _require_string(
                        artifact_status_values[artifact_id],
                        f"artifact {artifact_id} status",
                    )
                )
            except ValueError as error:
                raise ValidationGateError("invalid artifact status") from error
            artifact_statuses[artifact_id] = artifact_status
            digest = artifact_hashes[artifact_id]
            if artifact_status is ArtifactStatus.HASHED:
                _require_sha256(digest, f"artifact {artifact_id} SHA-256")
            elif digest is not None:
                raise ValidationGateError("unavailable artifact cannot have a hash")
        unexpected_artifact_count = _require_integer(
            audit["unexpected_artifact_count"],
            "unexpected artifact count",
        )

        derived_failures: set[IntegrityFailure] = set()
        if reported_integrity is IntegrityStatus.FAILED:
            derived_failures.add(IntegrityFailure.REPORTED_INTEGRITY_FAILED)
        if harbor_retry_count or harness_retry_count:
            derived_failures.add(IntegrityFailure.RETRY_COUNT_NONZERO)
        if repository_state is RepositoryState.UNAVAILABLE:
            derived_failures.add(IntegrityFailure.REPOSITORY_OBSERVATION_UNAVAILABLE)
        else:
            if observed_revision != expected_identity["repository_revision"]:
                derived_failures.add(IntegrityFailure.REPOSITORY_REVISION_MISMATCH)
            if repository_state is RepositoryState.DIRTY:
                derived_failures.add(IntegrityFailure.REPOSITORY_WORKTREE_DIRTY)
        if (
            observed_digests["declaration_sha256"]
            != expected_identity["declaration_sha256"]
        ):
            derived_failures.add(IntegrityFailure.DECLARATION_MISMATCH)
        manifest_matches = (
            observed_digests["task_manifest_sha256"]
            == expected_identity["task_manifest_sha256"]
        )
        if not manifest_matches:
            derived_failures.add(IntegrityFailure.TASK_MANIFEST_MISMATCH)
            if task_package_status is not RepositoryState.UNAVAILABLE:
                raise ValidationGateError("task package evidence is inconsistent")
        elif task_package_status is RepositoryState.UNAVAILABLE:
            derived_failures.add(IntegrityFailure.TASK_MANIFEST_CONTRACT_MISMATCH)
        elif task_package_status is IntegrityStatus.FAILED:
            derived_failures.add(IntegrityFailure.TASK_PACKAGE_MISMATCH)
        if (
            observed_digests["configuration_sha256"]
            != expected_identity["configuration_sha256"]
        ):
            derived_failures.add(IntegrityFailure.CONFIGURATION_MISMATCH)
        for artifact_status in artifact_statuses.values():
            if artifact_status is ArtifactStatus.MISSING:
                derived_failures.add(IntegrityFailure.REQUIRED_ARTIFACT_MISSING)
            elif artifact_status is ArtifactStatus.UNREADABLE:
                derived_failures.add(IntegrityFailure.ARTIFACT_UNREADABLE)
        if unexpected_artifact_count:
            derived_failures.add(IntegrityFailure.UNEXPECTED_ARTIFACT)

        if failures != derived_failures:
            raise ValidationGateError("integrity failures contradict retained evidence")
        expected_integrity_status = (
            IntegrityStatus.FAILED if derived_failures else IntegrityStatus.PASSED
        )
        if integrity_status is not expected_integrity_status:
            raise ValidationGateError("integrity status contradicts retained evidence")

    def _later_state_exists(self, start_index: int) -> bool:
        for slot in self.definition.schedule[start_index:]:
            if self.receipt_path(slot.slot_id).exists():
                return True
            if self.audit_path(slot.slot_id).exists():
                return True
        return False

    def snapshot(self) -> GateSnapshot:
        completed: list[CompletedSlot] = []
        for index, slot in enumerate(self.definition.schedule):
            receipt_path = self.receipt_path(slot.slot_id)
            audit_path = self.audit_path(slot.slot_id)
            if not receipt_path.exists():
                if audit_path.exists() or self._later_state_exists(index + 1):
                    raise ValidationGateError("gate state is not a schedule prefix")
                return self._snapshot_after_completed(completed)
            receipt, receipt_sha256 = self._read_receipt(slot)
            if not audit_path.exists():
                if self._later_state_exists(index + 1):
                    raise ValidationGateError("gate state advances past an active slot")
                return GateSnapshot(
                    disposition=GateDisposition.IN_PROGRESS,
                    completed_slot_ids=tuple(item.slot_id for item in completed),
                    active_slot_id=slot.slot_id,
                )
            completed.append(self._read_terminal_audit(slot, receipt, receipt_sha256))
        return self._snapshot_after_completed(completed)

    def _snapshot_after_completed(
        self,
        completed: Sequence[CompletedSlot],
    ) -> GateSnapshot:
        completed_slot_ids = tuple(item.slot_id for item in completed)
        if any(
            item.terminal_outcome is TerminalOutcome.INFRASTRUCTURE_FAILURE
            for item in completed
        ):
            return GateSnapshot(
                disposition=GateDisposition.STOPPED,
                completed_slot_ids=completed_slot_ids,
                stop_reason=GateStopReason.INFRASTRUCTURE_FAILURE,
            )
        if any(item.integrity_status is IntegrityStatus.FAILED for item in completed):
            return GateSnapshot(
                disposition=GateDisposition.STOPPED,
                completed_slot_ids=completed_slot_ids,
                stop_reason=GateStopReason.INTEGRITY_FAILURE,
            )
        if len(completed) == len(self.definition.schedule):
            return GateSnapshot(
                disposition=GateDisposition.COMPLETE,
                completed_slot_ids=completed_slot_ids,
            )
        if len(completed) == 2:
            thorn = completed[1]
            if (
                thorn.terminal_outcome is not TerminalOutcome.COMPLETED
                or thorn.reward != 1.0
                or thorn.usability is not Usability.USABLE
            ):
                return GateSnapshot(
                    disposition=GateDisposition.STOPPED,
                    completed_slot_ids=completed_slot_ids,
                    stop_reason=GateStopReason.THORN_BOKEH_UNUSABLE,
                )
            tokens = [item.submitted_tokens for item in completed]
            if any(token is None for token in tokens):
                return GateSnapshot(
                    disposition=GateDisposition.STOPPED,
                    completed_slot_ids=completed_slot_ids,
                    stop_reason=GateStopReason.BOKEH_TOKENS_MISSING,
                )
            submitted_tokens = sum(cast(int, token) for token in tokens)
            if submitted_tokens > self.definition.between_block_submitted_token_limit:
                return GateSnapshot(
                    disposition=GateDisposition.STOPPED,
                    completed_slot_ids=completed_slot_ids,
                    stop_reason=GateStopReason.BOKEH_TOKEN_LIMIT,
                )
        next_slot = self.definition.schedule[len(completed)]
        return GateSnapshot(
            disposition=GateDisposition.READY,
            completed_slot_ids=completed_slot_ids,
            next_slot_id=next_slot.slot_id,
        )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--expected-definition-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce immutable validation-v2 receipts and audits",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    status_parser = subparsers.add_parser("status")
    consume_parser = subparsers.add_parser("consume")
    record_parser = subparsers.add_parser("record-terminal")
    for command_parser in (
        validate_parser,
        status_parser,
        consume_parser,
        record_parser,
    ):
        _add_common_arguments(command_parser)
    consume_parser.add_argument("--authorization", type=Path, required=True)
    consume_parser.add_argument("--slot-id", required=True)
    consume_parser.add_argument("--confirm", required=True)
    record_parser.add_argument("--observation", type=Path, required=True)
    record_parser.add_argument("--confirm", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    controller = ValidationGateController.load(
        definition_path=arguments.definition,
        expected_definition_sha256=arguments.expected_definition_sha256,
        state_root=arguments.state_root,
        repository_probe=GitRepositoryProbe(arguments.repository_root),
    )
    try:
        if arguments.command == "validate":
            controller.validate_static_identities()
            output: dict[str, object] = {"status": "validated"}
        elif arguments.command == "status":
            output = controller.snapshot().as_json()
        elif arguments.command == "consume":
            receipt_sha256 = controller.consume_slot(
                slot_id=arguments.slot_id,
                authorization_path=arguments.authorization,
                confirmation=arguments.confirm,
            )
            output = {"receipt_sha256": receipt_sha256}
        else:
            observation = load_terminal_observation(arguments.observation)
            terminal_audit_sha256 = controller.record_terminal(
                observation,
                confirmation=arguments.confirm,
            )
            output = {"terminal_audit_sha256": terminal_audit_sha256}
    except ValidationGateError as error:
        raise SystemExit(f"validation gate refused operation: {error}") from error
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
