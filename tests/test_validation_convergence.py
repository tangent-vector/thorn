"""Tests for normalized validation identity and convergence decisions."""

from __future__ import annotations

import json

import pytest

from thorn.core._validation_convergence import (
    ValidationActionEpochReason,
    ValidationActionTool,
    ValidationCommand,
    ValidationConvergenceDecision,
    ValidationConvergenceObservation,
    ValidationConvergencePolicy,
    ValidationConvergenceTracker,
    ValidationOutcome,
    ValidationProgressEffect,
    ValidationRunner,
    ValidationScope,
    WorkspaceContentConvergenceDecision,
    WorkspaceContentConvergencePolicy,
    WorkspaceContentConvergenceTracker,
    parse_validation_command,
    validation_outcome_from_tool_result,
)
from thorn.core._workspace_content import (
    WorkspaceContentCollectionFailure,
    WorkspaceContentIdentity,
    WorkspaceContentSnapshot,
)


def _shell_arguments(
    command: str,
    *,
    working_directory: str | None = None,
) -> str:
    arguments: dict[str, str] = {"command": command}
    if working_directory is not None:
        arguments["working_directory"] = working_directory
    return json.dumps(arguments)


def _command(command: str) -> ValidationCommand:
    parsed = parse_validation_command(_shell_arguments(command))
    assert parsed is not None
    return parsed


def _observe(
    tracker: ValidationConvergenceTracker,
    command: str,
    outcome: ValidationOutcome,
    *,
    call_id: str = "validation-call",
    render_id: str = "provider-render",
    policy: ValidationConvergencePolicy = (
        ValidationConvergencePolicy.ACTION_EPOCH_V1
    ),
) -> ValidationConvergenceObservation:
    return tracker.observe(
        _command(command),
        outcome,
        call_id=call_id,
        render_id=render_id,
        policy=policy,
    )


class TestValidationCommandNormalization:
    def test_uv_and_python_wrappers_normalize_to_same_pytest_identity(self):
        uv_command = _command("uv run pytest -q tests/test_loop.py")
        python_command = _command(
            "python -m pytest --color=no tests/test_loop.py",
        )

        assert uv_command.identity == python_command.identity
        assert uv_command.identity.runner is ValidationRunner.PYTEST
        assert uv_command.identity.scope is ValidationScope.TARGETED

    def test_working_directory_is_part_of_identity(self):
        first = parse_validation_command(
            _shell_arguments("pytest", working_directory="project-a"),
        )
        second = parse_validation_command(
            _shell_arguments("pytest", working_directory="project-b"),
        )

        assert first is not None
        assert second is not None
        assert first.identity != second.identity

    def test_default_and_dot_working_directories_are_equivalent(self):
        implicit = parse_validation_command(_shell_arguments("pytest"))
        explicit = parse_validation_command(
            _shell_arguments("pytest", working_directory="."),
        )

        assert implicit is not None
        assert explicit is not None
        assert implicit.identity == explicit.identity

    def test_full_pytest_is_broader_than_targeted_pytest(self):
        targeted = _command("pytest tests/test_loop.py")
        full = _command("pytest")

        assert targeted.identity.scope is ValidationScope.TARGETED
        assert full.identity.scope is ValidationScope.WORKSPACE

    def test_default_and_dot_pytest_targets_are_equivalent(self):
        assert _command("pytest").identity == _command("pytest .").identity

    def test_mixed_shell_command_is_not_deduplicated(self):
        assert (
            parse_validation_command(
                _shell_arguments("uv run pytest && echo finished"),
            )
            is None
        )

    @pytest.mark.parametrize(
        "command",
        [
            "pytest > result.txt",
            "pytest 2>&1",
            "pytest $(touch marker)",
            "pytest <(generate-tests)",
            "pytest `touch marker`",
            "pytest | tee result.txt",
            "pytest; touch marker",
            "pytest *.py",
            "pytest # ignored shell comment",
        ],
    )
    def test_shell_composition_is_not_deduplicated(self, command: str):
        assert parse_validation_command(_shell_arguments(command)) is None

    def test_mutating_ruff_commands_are_not_validation(self):
        assert (
            parse_validation_command(
                _shell_arguments("ruff check --fix ."),
            )
            is None
        )
        assert (
            parse_validation_command(
                _shell_arguments("ruff format ."),
            )
            is None
        )


class TestValidationConvergenceTracker:
    def test_equivalent_success_only_counts_once_per_action_epoch(self):
        tracker = ValidationConvergenceTracker()

        first = _observe(
            tracker,
            "uv run pytest -q tests/test_loop.py",
            ValidationOutcome.SUCCEEDED,
            call_id="first-validation",
        )
        repeat = _observe(
            tracker,
            "python -m pytest --color=no tests/test_loop.py",
            ValidationOutcome.SUCCEEDED,
            call_id="repeat-validation",
        )

        assert (
            first.telemetry.decision
            is ValidationConvergenceDecision.FIRST_IN_ACTION_EPOCH
        )
        assert first.progress_effect is ValidationProgressEffect.COUNTS_AS_PROGRESS
        assert (
            repeat.telemetry.decision is ValidationConvergenceDecision.EQUIVALENT_REPEAT
        )
        assert (
            repeat.progress_effect
            is ValidationProgressEffect.DOES_NOT_COUNT_AS_PROGRESS
        )
        assert repeat.telemetry.prior_outcome is ValidationOutcome.SUCCEEDED
        assert repeat.telemetry.call_id == "repeat-validation"
        assert repeat.telemetry.render_id == "provider-render"
        assert repeat.telemetry.policy_effect_applied is True

    def test_material_action_starts_fresh_validation_epoch(self):
        tracker = ValidationConvergenceTracker()
        _observe(
            tracker,
            "pytest tests/test_loop.py",
            ValidationOutcome.SUCCEEDED,
        )

        advance = tracker.advance_workspace_action_epoch(
            call_id="edit-call",
            render_id="edit-render",
            reason=ValidationActionEpochReason.NATIVE_MATERIAL_MUTATION,
            tool_name=ValidationActionTool.EDIT_FILE,
        )
        after_edit = _observe(
            tracker,
            "pytest tests/test_loop.py",
            ValidationOutcome.SUCCEEDED,
        )

        assert advance.to_json() == {
            "telemetry_schema_version": 2,
            "call_id": "edit-call",
            "render_id": "edit-render",
            "prior_workspace_action_epoch": 0,
            "workspace_action_epoch": 1,
            "reason": "native_material_mutation",
            "tool_name": "edit_file",
        }
        assert after_edit.telemetry.workspace_action_epoch.value == 1
        assert (
            after_edit.telemetry.decision
            is ValidationConvergenceDecision.FIRST_IN_ACTION_EPOCH
        )

    def test_changed_outcome_counts_as_progress(self):
        tracker = ValidationConvergenceTracker()
        _observe(
            tracker,
            "pytest tests/test_loop.py",
            ValidationOutcome.FAILED,
        )

        recovered = _observe(
            tracker,
            "pytest tests/test_loop.py",
            ValidationOutcome.SUCCEEDED,
        )

        assert (
            recovered.telemetry.decision
            is ValidationConvergenceDecision.CHANGED_OUTCOME
        )
        assert recovered.telemetry.prior_outcome is ValidationOutcome.FAILED
        assert recovered.progress_effect is ValidationProgressEffect.COUNTS_AS_PROGRESS

    def test_broader_validation_counts_as_distinct_progress(self):
        tracker = ValidationConvergenceTracker()
        _observe(
            tracker,
            "pytest tests/test_loop.py",
            ValidationOutcome.SUCCEEDED,
        )

        broader = _observe(
            tracker,
            "pytest",
            ValidationOutcome.SUCCEEDED,
        )

        assert (
            broader.telemetry.decision
            is ValidationConvergenceDecision.BROADER_VALIDATION
        )
        assert broader.progress_effect is ValidationProgressEffect.COUNTS_AS_PROGRESS

    def test_indeterminate_outcome_defers_to_conservative_classifier(self):
        tracker = ValidationConvergenceTracker()

        observation = _observe(
            tracker,
            "pytest",
            ValidationOutcome.INDETERMINATE,
        )

        assert (
            observation.telemetry.decision
            is ValidationConvergenceDecision.INDETERMINATE_OUTCOME
        )
        assert (
            observation.progress_effect
            is ValidationProgressEffect.DEFER_TO_CONSERVATIVE_CLASSIFIER
        )

    def test_baseline_records_counterfactual_effect_without_applying_it(self):
        observation = _observe(
            ValidationConvergenceTracker(),
            "pytest",
            ValidationOutcome.SUCCEEDED,
            policy=ValidationConvergencePolicy.BASELINE,
        )

        assert (
            observation.progress_effect
            is ValidationProgressEffect.COUNTS_AS_PROGRESS
        )
        assert observation.telemetry.policy_effect_applied is False
        assert observation.telemetry.to_json()["telemetry_schema_version"] == 2


class TestValidationOutcomeNormalization:
    def test_exit_code_prefix_is_failure(self):
        assert (
            validation_outcome_from_tool_result(
                content="[exit code 1]\nthree failures",
                tool_reported_error=False,
            )
            is ValidationOutcome.FAILED
        )

    def test_negative_exit_code_prefix_is_failure(self):
        assert (
            validation_outcome_from_tool_result(
                content="[exit code -9]\nterminated by signal",
                tool_reported_error=False,
            )
            is ValidationOutcome.FAILED
        )

    def test_tool_error_and_timeout_are_indeterminate(self):
        assert (
            validation_outcome_from_tool_result(
                content="tool crashed",
                tool_reported_error=True,
            )
            is ValidationOutcome.INDETERMINATE
        )
        assert (
            validation_outcome_from_tool_result(
                content="[timed out after 120s]\npartial output",
                tool_reported_error=False,
            )
            is ValidationOutcome.INDETERMINATE
        )

    def test_telemetry_json_contains_no_command_or_output(self):
        tracker = ValidationConvergenceTracker()
        observation = _observe(
            tracker,
            "pytest secret/customer_case.py",
            ValidationOutcome.SUCCEEDED,
        )

        encoded = json.dumps(observation.telemetry.to_json())

        assert "secret" not in encoded
        assert "customer_case" not in encoded
        assert "command" not in encoded
        assert "output" not in encoded
        assert len(observation.telemetry.identity_fingerprint) == 64


class TestWorkspaceContentConvergenceTracker:
    @staticmethod
    def _known(identity: str) -> WorkspaceContentSnapshot:
        return WorkspaceContentSnapshot.known(
            identity=WorkspaceContentIdentity(identity),
            path_count=3,
            content_bytes=120,
        )

    def test_opaque_actions_do_not_hide_equivalent_validation_on_same_content(self):
        tracker = WorkspaceContentConvergenceTracker()
        command = _command("pytest tests/test_loop.py")

        first = tracker.observe(
            command,
            ValidationOutcome.SUCCEEDED,
            self._known("a" * 64),
            call_id="first",
            render_id="render-1",
            policy=WorkspaceContentConvergencePolicy.OBSERVE_V2,
        )
        repeat = tracker.observe(
            command,
            ValidationOutcome.SUCCEEDED,
            self._known("a" * 64),
            call_id="repeat",
            render_id="render-3",
            policy=WorkspaceContentConvergencePolicy.APPLY_V2,
        )

        assert first.telemetry.content_epoch.value == 0
        assert (
            first.telemetry.decision
            is WorkspaceContentConvergenceDecision.FIRST_IN_CONTENT_EPOCH
        )
        assert (
            repeat.telemetry.decision
            is WorkspaceContentConvergenceDecision.EQUIVALENT_REPEAT
        )
        assert (
            repeat.progress_effect
            is ValidationProgressEffect.DOES_NOT_COUNT_AS_PROGRESS
        )
        assert repeat.telemetry.policy_effect_applied is True

    def test_changed_content_allows_same_validation_again(self):
        tracker = WorkspaceContentConvergenceTracker()
        command = _command("pytest")
        tracker.observe(
            command,
            ValidationOutcome.SUCCEEDED,
            self._known("a" * 64),
            call_id="first",
            render_id="render-1",
            policy=WorkspaceContentConvergencePolicy.APPLY_V2,
        )

        after_edit = tracker.observe(
            command,
            ValidationOutcome.SUCCEEDED,
            self._known("b" * 64),
            call_id="after-edit",
            render_id="render-2",
            policy=WorkspaceContentConvergencePolicy.APPLY_V2,
        )

        assert after_edit.telemetry.content_epoch.value == 1
        assert (
            after_edit.telemetry.decision
            is WorkspaceContentConvergenceDecision.FIRST_IN_CONTENT_EPOCH
        )
        assert after_edit.progress_effect is ValidationProgressEffect.COUNTS_AS_PROGRESS

    def test_unknown_content_defers_without_forgetting_last_known_content(self):
        tracker = WorkspaceContentConvergenceTracker()
        command = _command("pytest")
        tracker.observe(
            command,
            ValidationOutcome.SUCCEEDED,
            self._known("a" * 64),
            call_id="first",
            render_id="render-1",
            policy=WorkspaceContentConvergencePolicy.APPLY_V2,
        )
        unknown = tracker.observe(
            command,
            ValidationOutcome.SUCCEEDED,
            WorkspaceContentSnapshot.unknown(
                WorkspaceContentCollectionFailure.CONCURRENT_CHANGE,
            ),
            call_id="unknown",
            render_id="render-2",
            policy=WorkspaceContentConvergencePolicy.APPLY_V2,
        )
        repeat = tracker.observe(
            command,
            ValidationOutcome.SUCCEEDED,
            self._known("a" * 64),
            call_id="repeat",
            render_id="render-3",
            policy=WorkspaceContentConvergencePolicy.APPLY_V2,
        )

        assert unknown.progress_effect is (
            ValidationProgressEffect.DEFER_TO_CONSERVATIVE_CLASSIFIER
        )
        assert unknown.telemetry.collection_failure == (
            WorkspaceContentCollectionFailure.CONCURRENT_CHANGE
        )
        assert (
            repeat.telemetry.decision
            is WorkspaceContentConvergenceDecision.EQUIVALENT_REPEAT
        )
