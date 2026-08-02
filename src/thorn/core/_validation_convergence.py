"""Stateful convergence decisions for repeated shell validations.

The no-progress guard needs to distinguish useful validation from a model
re-running the same check against unchanged state.  This module deliberately
recognizes only simple, well-understood validation commands.  The loop keeps
ambiguous shell commands on its conservative legacy path because treating an
unknown command as redundant could stop an agent after an unobserved shell edit.

Tracking is observational in every policy so experiment arms expose comparable
facts. The policy carried by each observation only decides whether the loop may
apply its recorded progress effect.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePath
from typing import Any

from thorn.core._workspace_content import (
    WorkspaceContentCollectionFailure,
    WorkspaceContentIdentity,
    WorkspaceContentSnapshot,
)


class ValidationRunner(StrEnum):
    """Validation command families understood by the convergence tracker."""

    PYTEST = "pytest"
    RUFF_CHECK = "ruff_check"
    RUFF_FORMAT_CHECK = "ruff_format_check"
    MYPY = "mypy"
    PYRIGHT = "pyright"
    NPM_TEST = "npm_test"
    PNPM_TEST = "pnpm_test"
    YARN_TEST = "yarn_test"
    CARGO_TEST = "cargo_test"
    GO_TEST = "go_test"


class ValidationConvergencePolicy(StrEnum):
    """Loop-progress policies for recognized validation commands."""

    BASELINE = "baseline"
    ACTION_EPOCH_V1 = "action-epoch-v1"
    WORKSPACE_CONTENT_OBSERVE_V2 = "workspace-content-observe-v2"
    WORKSPACE_CONTENT_V2 = "workspace-content-v2"

    @property
    def applies_progress_effect(self) -> bool:
        """Whether observations may change the legacy no-progress behavior."""
        return self in {
            ValidationConvergencePolicy.ACTION_EPOCH_V1,
            ValidationConvergencePolicy.WORKSPACE_CONTENT_V2,
        }

    @property
    def uses_workspace_content(self) -> bool:
        """Whether this arm collects and compares content snapshots."""
        return self in {
            ValidationConvergencePolicy.WORKSPACE_CONTENT_OBSERVE_V2,
            ValidationConvergencePolicy.WORKSPACE_CONTENT_V2,
        }

class WorkspaceContentConvergencePolicy(StrEnum):
    """Control and treatment effects for content-based convergence."""

    OBSERVE_V2 = "workspace-content-observe-v2"
    APPLY_V2 = "workspace-content-v2"

    @property
    def applies_progress_effect(self) -> bool:
        return self is WorkspaceContentConvergencePolicy.APPLY_V2

    @classmethod
    def from_validation_policy(
        cls,
        policy: ValidationConvergencePolicy,
    ) -> WorkspaceContentConvergencePolicy:
        if policy is ValidationConvergencePolicy.WORKSPACE_CONTENT_OBSERVE_V2:
            return cls.OBSERVE_V2
        if policy is ValidationConvergencePolicy.WORKSPACE_CONTENT_V2:
            return cls.APPLY_V2
        raise ValueError(f"policy does not use workspace content: {policy.value}")


class ValidationScope(StrEnum):
    """How much of a workspace a recognized command is known to validate."""

    TARGETED = "targeted"
    WORKSPACE = "workspace"
    UNKNOWN = "unknown"


class ValidationOutcome(StrEnum):
    """Normalized terminal result of a shell validation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


class ValidationConvergenceDecision(StrEnum):
    """Why an observed validation does or does not count as progress."""

    FIRST_IN_ACTION_EPOCH = "first_in_action_epoch"
    DISTINCT_VALIDATION = "distinct_validation"
    BROADER_VALIDATION = "broader_validation"
    CHANGED_OUTCOME = "changed_outcome"
    EQUIVALENT_REPEAT = "equivalent_repeat"
    INDETERMINATE_OUTCOME = "indeterminate_outcome"


class ValidationProgressEffect(StrEnum):
    """How a convergence decision affects the loop's no-progress counter."""

    COUNTS_AS_PROGRESS = "counts_as_progress"
    DOES_NOT_COUNT_AS_PROGRESS = "does_not_count_as_progress"
    DEFER_TO_CONSERVATIVE_CLASSIFIER = "defer_to_conservative_classifier"


class WorkspaceContentStateTransition(StrEnum):
    """Relationship between the current and last known task content."""

    KNOWN_SAME = "known_same"
    KNOWN_CHANGED = "known_changed"
    UNKNOWN = "unknown"


class WorkspaceContentConvergenceDecision(StrEnum):
    """Content-aware validation classification for schema-v3 trials."""

    FIRST_IN_CONTENT_EPOCH = "first_in_content_epoch"
    DISTINCT_VALIDATION = "distinct_validation"
    BROADER_VALIDATION = "broader_validation"
    CHANGED_OUTCOME = "changed_outcome"
    EQUIVALENT_REPEAT = "equivalent_repeat"
    INDETERMINATE_OUTCOME = "indeterminate_outcome"
    UNKNOWN_CONTENT = "unknown_content"


@dataclass(frozen=True, order=True)
class WorkspaceContentEpoch:
    """Monotonic epoch advanced only by observed content changes."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("workspace content epoch must be non-negative")

    def next(self) -> WorkspaceContentEpoch:
        return WorkspaceContentEpoch(self.value + 1)


class ValidationActionEpochReason(StrEnum):
    """Why the validation tracker advanced its workspace/action epoch."""

    NATIVE_MATERIAL_MUTATION = "native_material_mutation"
    OPAQUE_SHELL_POSSIBLE_MUTATION = "opaque_shell_possible_mutation"


class ValidationActionTool(StrEnum):
    """Tools whose successful calls can advance the validation action epoch."""

    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    DELETE_FILE = "delete_file"
    MOVE_FILE = "move_file"
    RUN_SHELL = "run_shell"


@dataclass(frozen=True, order=True)
class WorkspaceActionEpoch:
    """Monotonic identity for state after known or possible material actions."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("workspace action epoch must be non-negative")

    def next(self) -> WorkspaceActionEpoch:
        """Return the epoch after one more material or uncertain action."""
        return WorkspaceActionEpoch(self.value + 1)


@dataclass(frozen=True)
class ValidationIdentity:
    """Normalized semantic identity used only for in-memory comparisons."""

    runner: ValidationRunner
    arguments: tuple[str, ...]
    working_directory: str | None
    scope: ValidationScope

    @property
    def fingerprint(self) -> str:
        """Return a stable privacy-safe digest for telemetry and comparison."""
        payload = json.dumps(
            {
                "runner": self.runner.value,
                "arguments": self.arguments,
                "working_directory": self.working_directory,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ValidationCommand:
    """A shell command understood well enough for stateful deduplication."""

    identity: ValidationIdentity


@dataclass(frozen=True)
class ValidationTelemetry(ABC):
    """Base class for ordered, privacy-safe validation telemetry events."""

    telemetry_schema_version: int = field(default=2, init=False)

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Return the event fields without raw command or output text."""


@dataclass(frozen=True)
class ValidationActionEpochTelemetry(ValidationTelemetry):
    """Privacy-safe record of one workspace/action epoch advance."""

    call_id: str
    render_id: str
    prior_workspace_action_epoch: WorkspaceActionEpoch
    workspace_action_epoch: WorkspaceActionEpoch
    reason: ValidationActionEpochReason
    tool_name: ValidationActionTool

    def to_json(self) -> dict[str, Any]:
        """Return a JSON representation containing no command or output text."""
        return {
            "telemetry_schema_version": self.telemetry_schema_version,
            "call_id": self.call_id,
            "render_id": self.render_id,
            "prior_workspace_action_epoch": (
                self.prior_workspace_action_epoch.value
            ),
            "workspace_action_epoch": self.workspace_action_epoch.value,
            "reason": self.reason.value,
            "tool_name": self.tool_name.value,
        }


@dataclass(frozen=True)
class ValidationConvergenceTelemetry(ValidationTelemetry):
    """Privacy-safe facts about one recognized validation decision."""

    call_id: str
    render_id: str
    workspace_action_epoch: WorkspaceActionEpoch
    runner: ValidationRunner
    identity_fingerprint: str
    scope: ValidationScope
    outcome: ValidationOutcome
    prior_outcome: ValidationOutcome | None
    decision: ValidationConvergenceDecision
    progress_effect: ValidationProgressEffect
    policy_effect_applied: bool

    def to_json(self) -> dict[str, Any]:
        """Return a JSON representation containing no command or output text."""
        return {
            "telemetry_schema_version": self.telemetry_schema_version,
            "call_id": self.call_id,
            "render_id": self.render_id,
            "workspace_action_epoch": self.workspace_action_epoch.value,
            "runner": self.runner.value,
            "identity_fingerprint": self.identity_fingerprint,
            "validation_scope": self.scope.value,
            "outcome": self.outcome.value,
            "prior_outcome": (
                self.prior_outcome.value if self.prior_outcome is not None else None
            ),
            "decision": self.decision.value,
            "progress_effect": self.progress_effect.value,
            "policy_effect_applied": self.policy_effect_applied,
        }


@dataclass(frozen=True)
class ValidationConvergenceObservation:
    """One internal decision together with its safe telemetry record."""

    telemetry: ValidationConvergenceTelemetry

    @property
    def progress_effect(self) -> ValidationProgressEffect:
        return self.telemetry.progress_effect


@dataclass
class ValidationConvergenceTracker:
    """Track validation outcomes within the current workspace/action epoch."""

    workspace_action_epoch: WorkspaceActionEpoch = field(
        default_factory=lambda: WorkspaceActionEpoch(0),
    )
    _outcomes: dict[ValidationIdentity, ValidationOutcome] = field(
        default_factory=dict,
    )

    def advance_workspace_action_epoch(
        self,
        *,
        call_id: str,
        render_id: str,
        reason: ValidationActionEpochReason,
        tool_name: ValidationActionTool,
    ) -> ValidationActionEpochTelemetry:
        """Forget prior validations and describe the action that caused it."""
        prior_workspace_action_epoch = self.workspace_action_epoch
        self.workspace_action_epoch = prior_workspace_action_epoch.next()
        self._outcomes.clear()
        return ValidationActionEpochTelemetry(
            call_id=call_id,
            render_id=render_id,
            prior_workspace_action_epoch=prior_workspace_action_epoch,
            workspace_action_epoch=self.workspace_action_epoch,
            reason=reason,
            tool_name=tool_name,
        )

    def observe(
        self,
        command: ValidationCommand,
        outcome: ValidationOutcome,
        *,
        call_id: str,
        render_id: str,
        policy: ValidationConvergencePolicy,
    ) -> ValidationConvergenceObservation:
        """Classify and remember one validation in the current action epoch."""
        identity = command.identity
        prior_outcome = self._outcomes.get(identity)

        if outcome is ValidationOutcome.INDETERMINATE:
            return self._observation(
                identity=identity,
                outcome=outcome,
                prior_outcome=prior_outcome,
                decision=ValidationConvergenceDecision.INDETERMINATE_OUTCOME,
                progress_effect=(
                    ValidationProgressEffect.DEFER_TO_CONSERVATIVE_CLASSIFIER
                ),
                call_id=call_id,
                render_id=render_id,
                policy=policy,
            )

        if prior_outcome is not None:
            self._outcomes[identity] = outcome
            if prior_outcome is outcome:
                return self._observation(
                    identity=identity,
                    outcome=outcome,
                    prior_outcome=prior_outcome,
                    decision=ValidationConvergenceDecision.EQUIVALENT_REPEAT,
                    progress_effect=(
                        ValidationProgressEffect.DOES_NOT_COUNT_AS_PROGRESS
                    ),
                    call_id=call_id,
                    render_id=render_id,
                    policy=policy,
                )
            return self._observation(
                identity=identity,
                outcome=outcome,
                prior_outcome=prior_outcome,
                decision=ValidationConvergenceDecision.CHANGED_OUTCOME,
                progress_effect=ValidationProgressEffect.COUNTS_AS_PROGRESS,
                call_id=call_id,
                render_id=render_id,
                policy=policy,
            )

        previous_identities = tuple(self._outcomes)
        self._outcomes[identity] = outcome
        if not previous_identities:
            decision = ValidationConvergenceDecision.FIRST_IN_ACTION_EPOCH
        elif _is_broader_than_prior(identity, previous_identities):
            decision = ValidationConvergenceDecision.BROADER_VALIDATION
        else:
            decision = ValidationConvergenceDecision.DISTINCT_VALIDATION
        return self._observation(
            identity=identity,
            outcome=outcome,
            prior_outcome=None,
            decision=decision,
            progress_effect=ValidationProgressEffect.COUNTS_AS_PROGRESS,
            call_id=call_id,
            render_id=render_id,
            policy=policy,
        )

    def _observation(
        self,
        *,
        identity: ValidationIdentity,
        outcome: ValidationOutcome,
        prior_outcome: ValidationOutcome | None,
        decision: ValidationConvergenceDecision,
        progress_effect: ValidationProgressEffect,
        call_id: str,
        render_id: str,
        policy: ValidationConvergencePolicy,
    ) -> ValidationConvergenceObservation:
        return ValidationConvergenceObservation(
            telemetry=ValidationConvergenceTelemetry(
                call_id=call_id,
                render_id=render_id,
                workspace_action_epoch=self.workspace_action_epoch,
                runner=identity.runner,
                identity_fingerprint=identity.fingerprint,
                scope=identity.scope,
                outcome=outcome,
                prior_outcome=prior_outcome,
                decision=decision,
                progress_effect=progress_effect,
                policy_effect_applied=policy.applies_progress_effect,
            ),
        )


@dataclass(frozen=True)
class WorkspaceContentConvergenceTelemetry(ValidationTelemetry):
    """Schema-v3 content-aware validation facts without raw task data."""

    call_id: str
    render_id: str
    content_epoch: WorkspaceContentEpoch
    content_identity: WorkspaceContentIdentity | None
    prior_content_identity: WorkspaceContentIdentity | None
    content_transition: WorkspaceContentStateTransition
    collection_failure: WorkspaceContentCollectionFailure | None
    path_count: int | None
    content_bytes: int | None
    runner: ValidationRunner
    identity_fingerprint: str
    scope: ValidationScope
    outcome: ValidationOutcome
    prior_outcome: ValidationOutcome | None
    decision: WorkspaceContentConvergenceDecision
    progress_effect: ValidationProgressEffect
    policy_effect_applied: bool
    telemetry_schema_version: int = field(default=3, init=False)

    def to_json(self) -> dict[str, Any]:
        """Return privacy-safe content equality and convergence evidence."""
        return {
            "telemetry_schema_version": self.telemetry_schema_version,
            "call_id": self.call_id,
            "render_id": self.render_id,
            "workspace_content_epoch": self.content_epoch.value,
            "workspace_content_identity": (
                self.content_identity.value
                if self.content_identity is not None
                else None
            ),
            "prior_workspace_content_identity": (
                self.prior_content_identity.value
                if self.prior_content_identity is not None
                else None
            ),
            "workspace_content_transition": self.content_transition.value,
            "workspace_content_collection_failure": (
                self.collection_failure.value
                if self.collection_failure is not None
                else None
            ),
            "workspace_content_path_count": self.path_count,
            "workspace_content_bytes": self.content_bytes,
            "runner": self.runner.value,
            "identity_fingerprint": self.identity_fingerprint,
            "validation_scope": self.scope.value,
            "outcome": self.outcome.value,
            "prior_outcome": (
                self.prior_outcome.value if self.prior_outcome is not None else None
            ),
            "decision": self.decision.value,
            "progress_effect": self.progress_effect.value,
            "policy_effect_applied": self.policy_effect_applied,
        }


@dataclass(frozen=True)
class WorkspaceContentConvergenceObservation:
    """One content-aware decision together with its schema-v3 telemetry."""

    telemetry: WorkspaceContentConvergenceTelemetry

    @property
    def progress_effect(self) -> ValidationProgressEffect:
        return self.telemetry.progress_effect


@dataclass
class WorkspaceContentConvergenceTracker:
    """Compare validators only while task content is known to be identical."""

    content_epoch: WorkspaceContentEpoch = field(
        default_factory=lambda: WorkspaceContentEpoch(0),
    )
    _content_identity: WorkspaceContentIdentity | None = None
    _outcomes: dict[ValidationIdentity, ValidationOutcome] = field(
        default_factory=dict,
    )

    def observe(
        self,
        command: ValidationCommand,
        outcome: ValidationOutcome,
        snapshot: WorkspaceContentSnapshot,
        *,
        call_id: str,
        render_id: str,
        policy: WorkspaceContentConvergencePolicy,
    ) -> WorkspaceContentConvergenceObservation:
        """Classify one validator using its immediately preceding snapshot."""
        identity = command.identity
        prior_content_identity = self._content_identity
        if not snapshot.is_known:
            return self._observation(
                identity=identity,
                outcome=outcome,
                prior_outcome=None,
                snapshot=snapshot,
                prior_content_identity=prior_content_identity,
                transition=WorkspaceContentStateTransition.UNKNOWN,
                decision=WorkspaceContentConvergenceDecision.UNKNOWN_CONTENT,
                progress_effect=(
                    ValidationProgressEffect.DEFER_TO_CONSERVATIVE_CLASSIFIER
                ),
                call_id=call_id,
                render_id=render_id,
                policy=policy,
            )

        current_content_identity = snapshot.identity
        assert current_content_identity is not None
        if prior_content_identity is None:
            transition = WorkspaceContentStateTransition.KNOWN_CHANGED
            self._content_identity = current_content_identity
        elif current_content_identity != prior_content_identity:
            transition = WorkspaceContentStateTransition.KNOWN_CHANGED
            self.content_epoch = self.content_epoch.next()
            self._content_identity = current_content_identity
            self._outcomes.clear()
        else:
            transition = WorkspaceContentStateTransition.KNOWN_SAME

        prior_outcome = self._outcomes.get(identity)
        if outcome is ValidationOutcome.INDETERMINATE:
            decision = WorkspaceContentConvergenceDecision.INDETERMINATE_OUTCOME
            progress_effect = (
                ValidationProgressEffect.DEFER_TO_CONSERVATIVE_CLASSIFIER
            )
        elif prior_outcome is not None:
            self._outcomes[identity] = outcome
            if prior_outcome is outcome:
                decision = WorkspaceContentConvergenceDecision.EQUIVALENT_REPEAT
                progress_effect = ValidationProgressEffect.DOES_NOT_COUNT_AS_PROGRESS
            else:
                decision = WorkspaceContentConvergenceDecision.CHANGED_OUTCOME
                progress_effect = ValidationProgressEffect.COUNTS_AS_PROGRESS
        else:
            previous_identities = tuple(self._outcomes)
            self._outcomes[identity] = outcome
            if not previous_identities:
                decision = (
                    WorkspaceContentConvergenceDecision.FIRST_IN_CONTENT_EPOCH
                )
            elif _is_broader_than_prior(identity, previous_identities):
                decision = WorkspaceContentConvergenceDecision.BROADER_VALIDATION
            else:
                decision = WorkspaceContentConvergenceDecision.DISTINCT_VALIDATION
            progress_effect = ValidationProgressEffect.COUNTS_AS_PROGRESS

        return self._observation(
            identity=identity,
            outcome=outcome,
            prior_outcome=prior_outcome,
            snapshot=snapshot,
            prior_content_identity=prior_content_identity,
            transition=transition,
            decision=decision,
            progress_effect=progress_effect,
            call_id=call_id,
            render_id=render_id,
            policy=policy,
        )

    def _observation(
        self,
        *,
        identity: ValidationIdentity,
        outcome: ValidationOutcome,
        prior_outcome: ValidationOutcome | None,
        snapshot: WorkspaceContentSnapshot,
        prior_content_identity: WorkspaceContentIdentity | None,
        transition: WorkspaceContentStateTransition,
        decision: WorkspaceContentConvergenceDecision,
        progress_effect: ValidationProgressEffect,
        call_id: str,
        render_id: str,
        policy: WorkspaceContentConvergencePolicy,
    ) -> WorkspaceContentConvergenceObservation:
        return WorkspaceContentConvergenceObservation(
            telemetry=WorkspaceContentConvergenceTelemetry(
                call_id=call_id,
                render_id=render_id,
                content_epoch=self.content_epoch,
                content_identity=snapshot.identity,
                prior_content_identity=prior_content_identity,
                content_transition=transition,
                collection_failure=snapshot.failure,
                path_count=snapshot.path_count,
                content_bytes=snapshot.content_bytes,
                runner=identity.runner,
                identity_fingerprint=identity.fingerprint,
                scope=identity.scope,
                outcome=outcome,
                prior_outcome=prior_outcome,
                decision=decision,
                progress_effect=progress_effect,
                policy_effect_applied=policy.applies_progress_effect,
            ),
        )


ValidationConvergenceEvent = (
    ValidationConvergenceTelemetry | WorkspaceContentConvergenceTelemetry
)
"""Either versioned validation-convergence telemetry representation."""


_AMBIGUOUS_SHELL_SYNTAX = frozenset(
    "$`|&;()<>\n\r\0#*?[]{}~",
)
_PYTEST_PRESENTATION_ARGUMENTS = frozenset(
    {
        "-q",
        "--quiet",
        "-v",
        "--verbose",
        "--disable-warnings",
    }
)
_PYTEST_SELECTION_OPTIONS_WITH_VALUES = frozenset(
    {"-k", "-m", "--deselect", "--ignore", "--ignore-glob"},
)
_PYTEST_OPTIONS_WITH_VALUES = frozenset(
    {
        "--basetemp",
        "--confcutdir",
        "--override-ini",
        "--rootdir",
        "-o",
    }
)
_EXIT_CODE_PREFIX = re.compile(r"^\[exit code -?\d+\](?:\n|$)")


def parse_validation_command(arguments: str) -> ValidationCommand | None:
    """Parse a ``run_shell`` argument object when its command is unambiguous."""
    parsed = _parse_shell_tool_arguments(arguments)
    if parsed is None:
        return None
    command, working_directory = parsed
    if any(character in command for character in _AMBIGUOUS_SHELL_SYNTAX):
        return None
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return None
    if not argv:
        return None

    argv = _strip_environment_assignments(argv)
    argv = _strip_runner_wrapper(argv)
    if not argv:
        return None

    executable = PurePath(argv[0]).name.lower()
    runner_and_arguments = _recognized_runner(executable, argv[1:])
    if runner_and_arguments is None:
        return None
    runner, runner_arguments = runner_and_arguments
    normalized_arguments = _normalize_runner_arguments(
        runner,
        runner_arguments,
    )
    scope = _validation_scope(runner, normalized_arguments)
    return ValidationCommand(
        identity=ValidationIdentity(
            runner=runner,
            arguments=normalized_arguments,
            working_directory=_normalize_working_directory(working_directory),
            scope=scope,
        ),
    )


def validation_outcome_from_tool_result(
    *,
    content: str,
    tool_reported_error: bool,
) -> ValidationOutcome:
    """Normalize the built-in shell result convention without retaining output."""
    if tool_reported_error or content.startswith("[timed out after "):
        return ValidationOutcome.INDETERMINATE
    if _EXIT_CODE_PREFIX.match(content):
        return ValidationOutcome.FAILED
    return ValidationOutcome.SUCCEEDED


def _parse_shell_tool_arguments(
    arguments: str,
) -> tuple[str, str | None] | None:
    if not arguments:
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    command = parsed.get("command")
    if not isinstance(command, str):
        return None
    working_directory = parsed.get("working_directory")
    if working_directory is not None and not isinstance(working_directory, str):
        return None
    return command, working_directory


def _strip_environment_assignments(argv: list[str]) -> list[str]:
    remaining = list(argv)
    while remaining and _is_environment_assignment(remaining[0]):
        remaining.pop(0)
    return remaining


def _normalize_working_directory(working_directory: str | None) -> str | None:
    if not working_directory or working_directory == ".":
        return None
    return str(PurePath(working_directory))


def _is_environment_assignment(argument: str) -> bool:
    if "=" not in argument:
        return False
    name, _value = argument.split("=", 1)
    return bool(name) and name.replace("_", "a").isalnum() and not name[0].isdigit()


def _strip_runner_wrapper(argv: list[str]) -> list[str]:
    if len(argv) >= 2 and PurePath(argv[0]).name.lower() == "uv" and argv[1] == "run":
        return argv[2:]
    executable = PurePath(argv[0]).name.lower()
    if (
        len(argv) >= 3
        and executable in {"python", "python3", "python.exe"}
        and argv[1] == "-m"
    ):
        return argv[2:]
    return argv


def _recognized_runner(
    executable: str,
    arguments: list[str],
) -> tuple[ValidationRunner, list[str]] | None:
    if executable == "pytest":
        return ValidationRunner.PYTEST, arguments
    if executable == "ruff" and arguments[:1] == ["check"]:
        if "--fix" in arguments or "--fix-only" in arguments:
            return None
        return ValidationRunner.RUFF_CHECK, arguments[1:]
    if executable == "ruff" and arguments[:1] == ["format"]:
        if "--check" not in arguments:
            return None
        return ValidationRunner.RUFF_FORMAT_CHECK, arguments[1:]
    if executable == "mypy":
        return ValidationRunner.MYPY, arguments
    if executable == "pyright":
        return ValidationRunner.PYRIGHT, arguments
    if executable in {"npm", "pnpm", "yarn"}:
        remaining = list(arguments)
        if remaining[:1] == ["run"]:
            remaining = remaining[1:]
        if remaining[:1] != ["test"]:
            return None
        runner = {
            "npm": ValidationRunner.NPM_TEST,
            "pnpm": ValidationRunner.PNPM_TEST,
            "yarn": ValidationRunner.YARN_TEST,
        }[executable]
        return runner, remaining[1:]
    if executable == "cargo" and arguments[:1] == ["test"]:
        return ValidationRunner.CARGO_TEST, arguments[1:]
    if executable == "go" and arguments[:1] == ["test"]:
        return ValidationRunner.GO_TEST, arguments[1:]
    return None


def _normalize_runner_arguments(
    runner: ValidationRunner,
    arguments: list[str],
) -> tuple[str, ...]:
    normalized = tuple(arguments)
    if runner is ValidationRunner.PYTEST:
        normalized = tuple(
            argument
            for argument in arguments
            if argument not in _PYTEST_PRESENTATION_ARGUMENTS
            and not re.fullmatch(r"-[qv]{2,}", argument)
            and not argument.startswith("--color=")
            and not argument.startswith("--tb=")
        )
    if runner in {
        ValidationRunner.PYTEST,
        ValidationRunner.RUFF_CHECK,
        ValidationRunner.RUFF_FORMAT_CHECK,
        ValidationRunner.PYRIGHT,
    } and normalized[-1:] == (".",):
        return normalized[:-1]
    return normalized


def _validation_scope(
    runner: ValidationRunner,
    arguments: tuple[str, ...],
) -> ValidationScope:
    if runner is ValidationRunner.PYTEST:
        return _pytest_scope(arguments)
    if runner in {ValidationRunner.RUFF_CHECK, ValidationRunner.RUFF_FORMAT_CHECK}:
        targets = tuple(
            argument for argument in arguments if not argument.startswith("-")
        )
        if not targets or targets == (".",):
            return ValidationScope.WORKSPACE
        return ValidationScope.TARGETED
    if runner in {ValidationRunner.MYPY, ValidationRunner.PYRIGHT}:
        targets = tuple(
            argument for argument in arguments if not argument.startswith("-")
        )
        if not targets or targets == (".",):
            return ValidationScope.WORKSPACE
        return ValidationScope.TARGETED
    if runner in {
        ValidationRunner.NPM_TEST,
        ValidationRunner.PNPM_TEST,
        ValidationRunner.YARN_TEST,
    }:
        return ValidationScope.WORKSPACE if not arguments else ValidationScope.TARGETED
    if runner is ValidationRunner.CARGO_TEST:
        return (
            ValidationScope.WORKSPACE
            if not arguments or all(argument.startswith("-") for argument in arguments)
            else ValidationScope.TARGETED
        )
    if runner is ValidationRunner.GO_TEST:
        targets = tuple(
            argument for argument in arguments if not argument.startswith("-")
        )
        if not targets or targets == ("./...",):
            return ValidationScope.WORKSPACE
        return ValidationScope.TARGETED
    return ValidationScope.UNKNOWN


def _pytest_scope(arguments: tuple[str, ...]) -> ValidationScope:
    expect_option_value = False
    for argument in arguments:
        if expect_option_value:
            expect_option_value = False
            continue
        if argument in _PYTEST_SELECTION_OPTIONS_WITH_VALUES:
            return ValidationScope.TARGETED
        if argument in _PYTEST_OPTIONS_WITH_VALUES:
            expect_option_value = True
            continue
        if argument.startswith("-"):
            continue
        return ValidationScope.TARGETED
    return ValidationScope.WORKSPACE


def _is_broader_than_prior(
    identity: ValidationIdentity,
    prior_identities: tuple[ValidationIdentity, ...],
) -> bool:
    if identity.scope is not ValidationScope.WORKSPACE:
        return False
    return any(
        prior.runner is identity.runner and prior.scope is ValidationScope.TARGETED
        for prior in prior_identities
    )


__all__ = [
    "ValidationCommand",
    "ValidationActionEpochReason",
    "ValidationActionEpochTelemetry",
    "ValidationActionTool",
    "ValidationConvergenceDecision",
    "ValidationConvergenceEvent",
    "ValidationConvergenceObservation",
    "ValidationConvergencePolicy",
    "ValidationConvergenceTelemetry",
    "ValidationConvergenceTracker",
    "ValidationIdentity",
    "ValidationOutcome",
    "ValidationProgressEffect",
    "ValidationRunner",
    "ValidationScope",
    "ValidationTelemetry",
    "WorkspaceActionEpoch",
    "WorkspaceContentConvergenceDecision",
    "WorkspaceContentConvergenceObservation",
    "WorkspaceContentConvergencePolicy",
    "WorkspaceContentConvergenceTelemetry",
    "WorkspaceContentConvergenceTracker",
    "WorkspaceContentEpoch",
    "WorkspaceContentStateTransition",
    "parse_validation_command",
    "validation_outcome_from_tool_result",
]
