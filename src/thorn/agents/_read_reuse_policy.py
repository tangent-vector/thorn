"""Named read-reuse policies for controlled ``thorn run`` trials."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from thorn.core._read_file_history import (
    READ_FILE_REUSE_TELEMETRY_SCHEMA_VERSION,
    SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
    ReadFileReusePolicy,
)


class RunReadReusePolicy(StrEnum):
    """Session read-memory treatments available to ``thorn run``."""

    BASELINE = "baseline"
    SESSION_LEDGER_V1 = "session-ledger-v1"


DEFAULT_RUN_READ_REUSE_POLICY = RunReadReusePolicy.BASELINE
"""Read-reuse policy selected when ``--read-reuse-policy`` is omitted."""


@dataclass(frozen=True)
class RunReadReusePolicyDefinition:
    """Stable identity plus observational and model-facing policies."""

    policy: RunReadReusePolicy
    read_file_observation_policy: ReadFileReusePolicy
    read_file_advisory_policy: ReadFileReusePolicy | None

    def to_metadata(self) -> dict[str, str | int]:
        """Return the policy identity written to evaluation artifacts."""
        return {
            "read_reuse_policy": self.policy.value,
            "read_reuse_telemetry_schema_version": (
                READ_FILE_REUSE_TELEMETRY_SCHEMA_VERSION
            ),
        }


RUN_READ_REUSE_POLICY_DEFINITIONS: dict[
    RunReadReusePolicy,
    RunReadReusePolicyDefinition,
] = {
    RunReadReusePolicy.BASELINE: RunReadReusePolicyDefinition(
        policy=RunReadReusePolicy.BASELINE,
        read_file_observation_policy=(
            SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY
        ),
        read_file_advisory_policy=None,
    ),
    RunReadReusePolicy.SESSION_LEDGER_V1: RunReadReusePolicyDefinition(
        policy=RunReadReusePolicy.SESSION_LEDGER_V1,
        read_file_observation_policy=(
            SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY
        ),
        read_file_advisory_policy=(
            SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY
        ),
    ),
}


__all__ = [
    "DEFAULT_RUN_READ_REUSE_POLICY",
    "RUN_READ_REUSE_POLICY_DEFINITIONS",
    "RunReadReusePolicy",
    "RunReadReusePolicyDefinition",
]
