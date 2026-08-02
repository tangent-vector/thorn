from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext

from evals.harbor.thorn_harbor_agent import (
    READ_REUSE_TELEMETRY_SCHEMA_VERSION,
    ThornHarborAgent,
)


def _make_agent(
    tmp_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
    task_shell_environment: str = "inherit",
    action_policy: str = "baseline",
    history_policy: str = "baseline",
    validation_convergence_policy: str = "baseline",
    read_reuse_policy: str = "baseline",
    prompt_trace_capture: str = "redacted",
) -> ThornHarborAgent:
    wheel_path = tmp_path / "thorn-0.2.0-py3-none-any.whl"
    wheel_path.write_bytes(b"representative wheel bytes")
    constraints_path = tmp_path / "thorn-constraints.txt"
    constraints_path.write_text("httpx==0.28.1\n", encoding="utf-8")
    return ThornHarborAgent(
        logs_dir=tmp_path / "logs",
        thorn_wheel_path=str(wheel_path),
        thorn_constraints_path=str(constraints_path),
        thorn_revision="deadbeef",
        extra_env=extra_env,
        task_shell_environment=task_shell_environment,
        action_policy=action_policy,
        history_policy=history_policy,
        validation_convergence_policy=validation_convergence_policy,
        read_reuse_policy=read_reuse_policy,
        prompt_trace_capture=prompt_trace_capture,
    )


def _environment_with_result(return_code: int = 0) -> Any:
    environment = SimpleNamespace()
    environment.exec = AsyncMock(
        return_value=SimpleNamespace(
            return_code=return_code,
            stdout="thorn output",
            stderr="thorn error" if return_code else "",
        )
    )
    environment.upload_file = AsyncMock()
    return environment


def test_harbor_factory_imports_agent_with_required_provenance(
    tmp_path: Path,
) -> None:
    wheel_path = tmp_path / "thorn-0.2.0-py3-none-any.whl"
    wheel_path.write_bytes(b"representative wheel bytes")
    constraints_path = tmp_path / "thorn-constraints.txt"
    constraints_path.write_text("httpx==0.28.1\n", encoding="utf-8")

    agent = AgentFactory.create_agent_from_import_path(
        "evals.harbor.thorn_harbor_agent:ThornHarborAgent",
        logs_dir=tmp_path / "logs",
        thorn_wheel_path=str(wheel_path),
        thorn_constraints_path=str(constraints_path),
        thorn_revision="deadbeef",
    )

    assert isinstance(agent, ThornHarborAgent)
    assert agent.name() == "thorn"
    assert agent.version() == "deadbeef"
    assert agent.SUPPORTS_ATIF is False


def test_invalid_task_shell_environment_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="task_shell_environment must be one of",
    ):
        _make_agent(tmp_path, task_shell_environment="mystery-environment")


def test_invalid_action_policy_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="action_policy must be one of"):
        _make_agent(tmp_path, action_policy="untracked-experiment")


def test_invalid_history_policy_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="history_policy must be one of"):
        _make_agent(tmp_path, history_policy="untracked-experiment")


def test_invalid_validation_convergence_policy_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="validation_convergence_policy must be one of",
    ):
        _make_agent(
            tmp_path,
            validation_convergence_policy="untracked-experiment",
        )


def test_invalid_read_reuse_policy_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="read_reuse_policy must be one of"):
        _make_agent(tmp_path, read_reuse_policy="untracked-experiment")


def test_invalid_prompt_trace_capture_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt_trace_capture must be one of"):
        _make_agent(tmp_path, prompt_trace_capture="unsafe-ish")


def test_run_manifest_requires_read_reuse_telemetry_schema_version() -> None:
    harbor_directory = Path(__file__).parents[1]
    schema = json.loads(
        (harbor_directory / "run-manifest.schema.json").read_text(
            encoding="utf-8",
        )
    )
    example = json.loads(
        (harbor_directory / "run-manifest.example.json").read_text(
            encoding="utf-8",
        )
    )
    validator = Draft202012Validator(schema)

    example["agent"]["history_policy"] = "bounded-history-v2"
    example["agent"]["read_reuse_policy"] = "session-ledger-v1"
    validator.validate(example)
    assert example["schema_version"] == "1.7.0"
    assert example["agent"]["read_reuse_telemetry_schema_version"] == (
        READ_REUSE_TELEMETRY_SCHEMA_VERSION
    )

    del example["agent"]["read_reuse_telemetry_schema_version"]
    missing_field_errors = list(validator.iter_errors(example))
    assert any(
        error.validator == "required"
        and "read_reuse_telemetry_schema_version" in error.message
        for error in missing_field_errors
    )

    example["agent"]["read_reuse_telemetry_schema_version"] = 3
    wrong_version_errors = list(validator.iter_errors(example))
    assert any(
        error.validator == "const"
        and list(error.absolute_path) == [
            "agent",
            "read_reuse_telemetry_schema_version",
        ]
        for error in wrong_version_errors
    )


@pytest.mark.asyncio
async def test_install_uploads_wheel_and_pins_managed_toolchain(
    tmp_path: Path,
) -> None:
    agent = _make_agent(tmp_path)
    environment = _environment_with_result()

    await agent.install(cast(Any, environment))

    uploaded_targets = {
        call.args[1] for call in environment.upload_file.await_args_list
    }
    assert uploaded_targets == {
        "/installed-agent/thorn-0.2.0-py3-none-any.whl",
        "/installed-agent/thorn-constraints.txt",
        "/logs/agent/thorn-harbor-provenance.json",
    }

    install_call = environment.exec.await_args
    install_command = install_call.kwargs["command"]
    assert "https://astral.sh/uv/0.7.13/install.sh" in install_command
    assert "uv python install 3.11" in install_command
    assert "/installed-agent/thorn-venv" in install_command
    assert "--constraints /installed-agent/thorn-constraints.txt" in install_command
    assert install_call.kwargs["user"] == "root"

    manifest = json.loads(
        (agent.logs_dir / "setup" / "thorn-harbor-provenance.json").read_text()
    )
    assert manifest["harbor_revision"] == (
        "071281b3d931aafd6a5375fa7d5933e23054d784"
    )
    assert manifest["thorn_revision"] == "deadbeef"
    assert manifest["thorn_wheel"]["sha256"]
    assert manifest["thorn_constraints"]["sha256"]
    assert manifest["task_shell_environment"] == "inherit"
    assert manifest["action_policy"] == "baseline"
    assert manifest["history_policy"] == "baseline"
    assert manifest["validation_convergence_policy"] == "baseline"
    assert manifest["read_reuse_policy"] == "baseline"
    assert manifest["read_reuse_telemetry_schema_version"] == 2
    assert manifest["prompt_trace_capture"] == "redacted"
    assert manifest["supports_atif"] is False


@pytest.mark.asyncio
async def test_run_quotes_instruction_and_keeps_provider_secrets_out_of_command(
    tmp_path: Path,
) -> None:
    provider_secret = "provider-secret-with-'quotes'"
    agent = _make_agent(
        tmp_path,
        extra_env={
            "OPENAI_API_KEY": provider_secret,
            "OPENAI_API_URL": "https://provider.example/v1",
            "OPENAI_API_MODEL_NAME": "model-name",
        },
    )
    environment = _environment_with_result()
    instruction = "Fix the bug; echo 'this is still prompt text'"
    context = AgentContext()

    await agent.run(instruction, cast(Any, environment), context)

    run_call = environment.exec.await_args
    run_command = run_call.kwargs["command"]
    assert shlex.quote(instruction) in run_command
    assert provider_secret not in run_command
    assert "--agent-profile local" in run_command
    assert "--action-policy baseline" in run_command
    assert "--history-policy baseline" in run_command
    assert "--validation-convergence-policy baseline" in run_command
    assert "--read-reuse-policy baseline" in run_command
    assert "--workspace /testbed" in run_command
    assert "--agency \"$THORN_HARBOR_AGENCY_HOME\"" in run_command
    assert "--trace /logs/agent/thorn.jsonl" in run_command
    assert "--trace-raw-prompts" not in run_command
    assert "--result-file /logs/agent/thorn-result.json" in run_command
    assert "tee /logs/agent/thorn-output.txt" in run_command
    assert "conda activate" not in run_command
    assert run_call.kwargs["cwd"] == "/testbed"
    assert agent.extra_env["OPENAI_API_KEY"] == provider_secret
    assert context.is_empty()


@pytest.mark.asyncio
async def test_conda_testbed_environment_is_active_for_thorn_and_subprocesses(
    tmp_path: Path,
) -> None:
    agent = _make_agent(
        tmp_path,
        task_shell_environment="conda-testbed",
    )
    environment = _environment_with_result()

    await agent.install(cast(Any, environment))
    await agent.run(
        "Run the targeted tests",
        cast(Any, environment),
        AgentContext(),
    )

    run_command = environment.exec.await_args.kwargs["command"]
    activation_index = run_command.index(". /opt/miniconda3/bin/activate")
    thorn_index = run_command.index("/installed-agent/thorn-venv/bin/thorn run")
    assert activation_index < thorn_index
    assert "conda activate testbed" in run_command
    assert 'test "$CONDA_DEFAULT_ENV" = "testbed"' in run_command

    manifest = json.loads(
        (agent.logs_dir / "setup" / "thorn-harbor-provenance.json").read_text()
    )
    assert manifest["task_shell_environment"] == "conda-testbed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_policy",
    ["bounded-action-v1", "semantic-work-v2"],
)
async def test_action_policy_reaches_command_and_provenance(
    tmp_path: Path,
    action_policy: str,
) -> None:
    agent = _make_agent(tmp_path, action_policy=action_policy)
    environment = _environment_with_result()

    await agent.install(cast(Any, environment))
    await agent.run(
        "Make the requested change",
        cast(Any, environment),
        AgentContext(),
    )

    run_command = environment.exec.await_args.kwargs["command"]
    assert f"--action-policy {action_policy}" in run_command
    manifest = json.loads(
        (agent.logs_dir / "setup" / "thorn-harbor-provenance.json").read_text()
    )
    assert manifest["action_policy"] == action_policy


@pytest.mark.parametrize(
    "history_policy",
    ["bounded-history-v1", "bounded-history-v2"],
)
@pytest.mark.asyncio
async def test_bounded_history_policy_reaches_command_and_provenance(
    tmp_path: Path,
    history_policy: str,
) -> None:
    agent = _make_agent(tmp_path, history_policy=history_policy)
    environment = _environment_with_result()

    await agent.install(cast(Any, environment))
    await agent.run(
        "Make the requested change",
        cast(Any, environment),
        AgentContext(),
    )

    run_command = environment.exec.await_args.kwargs["command"]
    assert f"--history-policy {history_policy}" in run_command
    manifest = json.loads(
        (agent.logs_dir / "setup" / "thorn-harbor-provenance.json").read_text()
    )
    assert manifest["history_policy"] == history_policy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy",
    [
        "action-epoch-v1",
        "workspace-content-observe-v2",
        "workspace-content-v2",
    ],
)
async def test_validation_policy_reaches_command_and_provenance(
    tmp_path: Path,
    policy: str,
) -> None:
    agent = _make_agent(
        tmp_path,
        validation_convergence_policy=policy,
    )
    environment = _environment_with_result()

    await agent.install(cast(Any, environment))
    await agent.run(
        "Make the requested change",
        cast(Any, environment),
        AgentContext(),
    )

    run_command = environment.exec.await_args.kwargs["command"]
    assert f"--validation-convergence-policy {policy}" in run_command
    manifest = json.loads(
        (agent.logs_dir / "setup" / "thorn-harbor-provenance.json").read_text()
    )
    assert manifest["validation_convergence_policy"] == policy


@pytest.mark.asyncio
async def test_session_read_reuse_policy_reaches_command_and_provenance(
    tmp_path: Path,
) -> None:
    agent = _make_agent(tmp_path, read_reuse_policy="session-ledger-v1")
    environment = _environment_with_result()

    await agent.install(cast(Any, environment))
    await agent.run(
        "Make the requested change",
        cast(Any, environment),
        AgentContext(),
    )

    run_command = environment.exec.await_args.kwargs["command"]
    assert "--read-reuse-policy session-ledger-v1" in run_command
    manifest = json.loads(
        (agent.logs_dir / "setup" / "thorn-harbor-provenance.json").read_text()
    )
    assert manifest["read_reuse_policy"] == "session-ledger-v1"
    assert manifest["read_reuse_telemetry_schema_version"] == 2


@pytest.mark.asyncio
async def test_raw_prompt_capture_is_explicit_and_recorded(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, prompt_trace_capture="raw")
    environment = _environment_with_result()

    await agent.install(cast(Any, environment))
    await agent.run(
        "Audit the provider prompt",
        cast(Any, environment),
        AgentContext(),
    )

    run_command = environment.exec.await_args.kwargs["command"]
    assert "--trace-raw-prompts" in run_command
    manifest = json.loads(
        (agent.logs_dir / "setup" / "thorn-harbor-provenance.json").read_text()
    )
    assert manifest["prompt_trace_capture"] == "raw"


@pytest.mark.asyncio
async def test_run_preserves_nonzero_process_result(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    environment = _environment_with_result(return_code=7)

    with pytest.raises(NonZeroAgentExitCodeError):
        await agent.run(
            "Make the requested change",
            cast(Any, environment),
            AgentContext(),
        )

    assert environment.exec.await_args.kwargs["command"].startswith(
        "set -o pipefail;"
    )


def test_result_file_populates_harbor_context(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        action_policy="semantic-work-v2",
        history_policy="bounded-history-v1",
        validation_convergence_policy="action-epoch-v1",
        read_reuse_policy="session-ledger-v1",
    )
    agent.logs_dir.mkdir(parents=True)
    (agent.logs_dir / "thorn-result.json").write_text(
        json.dumps(
            {
                "outcome": "completed",
                "duration_s": 12.5,
                "token_usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 45,
                    "total_tokens": 168,
                },
                "error": None,
            }
        )
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 123
    assert context.n_output_tokens == 45
    assert context.metadata == {
        "thorn": {
            "profile": "local",
            "prompt_delivery": "direct",
            "action_policy": "semantic-work-v2",
            "history_policy": "bounded-history-v1",
            "validation_convergence_policy": "action-epoch-v1",
            "read_reuse_policy": "session-ledger-v1",
            "read_reuse_telemetry_schema_version": 2,
            "prompt_trace_capture": "redacted",
            "task_shell_environment": "inherit",
            "revision": "deadbeef",
            "result_file": "thorn-result.json",
            "trace_file": "thorn.jsonl",
            "outcome": "completed",
            "duration_s": 12.5,
            "error": None,
            "total_tokens": 168,
        }
    }
