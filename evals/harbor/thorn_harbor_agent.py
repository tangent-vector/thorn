"""Run a pinned Thorn wheel as a Harbor installed agent."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

HARBOR_REVISION = "071281b3d931aafd6a5375fa7d5933e23054d784"
UV_VERSION = "0.7.13"
PYTHON_VERSION = "3.11"
READ_REUSE_TELEMETRY_SCHEMA_VERSION = 2

_REMOTE_INSTALL_ROOT = PurePosixPath("/installed-agent")
_REMOTE_CONSTRAINTS_PATH = _REMOTE_INSTALL_ROOT / "thorn-constraints.txt"
_REMOTE_UV_DIRECTORY = _REMOTE_INSTALL_ROOT / "uv-bin"
_REMOTE_UV_PATH = _REMOTE_UV_DIRECTORY / "uv"
_REMOTE_PYTHON_DIRECTORY = _REMOTE_INSTALL_ROOT / "python"
_REMOTE_VENV_DIRECTORY = _REMOTE_INSTALL_ROOT / "thorn-venv"
_REMOTE_THORN_PATH = _REMOTE_VENV_DIRECTORY / "bin" / "thorn"

_REMOTE_WORKSPACE = PurePosixPath("/testbed")
_REMOTE_AGENT_LOGS = PurePosixPath("/logs/agent")
_TRACE_FILENAME = "thorn.jsonl"
_RESULT_FILENAME = "thorn-result.json"
_OUTPUT_FILENAME = "thorn-output.txt"
_PROVENANCE_FILENAME = "thorn-harbor-provenance.json"
_INSTALL_LOG_FILENAME = "thorn-install.txt"


class TaskShellEnvironment(StrEnum):
    """Task runtime environment inherited by Thorn subprocess tools."""

    INHERIT = "inherit"
    CONDA_TESTBED = "conda-testbed"

    @classmethod
    def parse(cls, raw_value: str) -> TaskShellEnvironment:
        try:
            return cls(raw_value.strip())
        except ValueError as exc:
            supported = ", ".join(value.value for value in cls)
            raise ValueError(
                f"task_shell_environment must be one of: {supported}",
            ) from exc


class ThornRunActionPolicy(StrEnum):
    """Named model-facing execution contract selected for the Thorn run."""

    BASELINE = "baseline"
    BOUNDED_ACTION_V1 = "bounded-action-v1"
    SEMANTIC_WORK_V2 = "semantic-work-v2"

    @classmethod
    def parse(cls, raw_value: str) -> ThornRunActionPolicy:
        try:
            return cls(raw_value.strip())
        except ValueError as exc:
            supported = ", ".join(value.value for value in cls)
            raise ValueError(
                f"action_policy must be one of: {supported}",
            ) from exc


class ThornRunHistoryPolicy(StrEnum):
    """Named provider-visible history projection selected for the Thorn run."""

    BASELINE = "baseline"
    BOUNDED_HISTORY_V1 = "bounded-history-v1"
    BOUNDED_HISTORY_V2 = "bounded-history-v2"

    @classmethod
    def parse(cls, raw_value: str) -> ThornRunHistoryPolicy:
        try:
            return cls(raw_value.strip())
        except ValueError as exc:
            supported = ", ".join(value.value for value in cls)
            raise ValueError(
                f"history_policy must be one of: {supported}",
            ) from exc


class ThornRunValidationConvergencePolicy(StrEnum):
    """Named validation-progress treatment selected for the Thorn run."""

    BASELINE = "baseline"
    ACTION_EPOCH_V1 = "action-epoch-v1"
    WORKSPACE_CONTENT_OBSERVE_V2 = "workspace-content-observe-v2"
    WORKSPACE_CONTENT_V2 = "workspace-content-v2"

    @classmethod
    def parse(
        cls,
        raw_value: str,
    ) -> ThornRunValidationConvergencePolicy:
        try:
            return cls(raw_value.strip())
        except ValueError as exc:
            supported = ", ".join(value.value for value in cls)
            raise ValueError(
                "validation_convergence_policy must be one of: "
                f"{supported}",
            ) from exc


class ThornRunReadReusePolicy(StrEnum):
    """Named session read-memory treatment selected for the Thorn run."""

    BASELINE = "baseline"
    SESSION_LEDGER_V1 = "session-ledger-v1"

    @classmethod
    def parse(cls, raw_value: str) -> ThornRunReadReusePolicy:
        try:
            return cls(raw_value.strip())
        except ValueError as exc:
            supported = ", ".join(value.value for value in cls)
            raise ValueError(
                f"read_reuse_policy must be one of: {supported}",
            ) from exc


class ThornPromptTraceCapture(StrEnum):
    """Sensitivity level for retained per-request prompt snapshots."""

    REDACTED = "redacted"
    RAW = "raw"

    @classmethod
    def parse(cls, raw_value: str) -> ThornPromptTraceCapture:
        try:
            return cls(raw_value.strip())
        except ValueError as exc:
            supported = ", ".join(value.value for value in cls)
            raise ValueError(
                f"prompt_trace_capture must be one of: {supported}",
            ) from exc


@dataclass(frozen=True)
class ThornRevision:
    """A non-empty source revision declared by an evaluation run."""

    value: str

    @classmethod
    def parse(cls, raw_revision: str) -> ThornRevision:
        revision = raw_revision.strip()
        if not revision:
            raise ValueError("thorn_revision must not be empty")
        return cls(revision)


@dataclass(frozen=True)
class ThornWheel:
    """A validated host wheel and its content identity."""

    path: Path
    sha256: str

    @classmethod
    def from_host_path(cls, raw_path: str | Path) -> ThornWheel:
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Thorn wheel path is not a file: {path}")
        if path.suffix != ".whl":
            raise ValueError(f"Thorn wheel must end in .whl: {path}")

        digest = hashlib.sha256()
        with path.open("rb") as wheel_file:
            for chunk in iter(lambda: wheel_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(path=path, sha256=digest.hexdigest())


@dataclass(frozen=True)
class ThornConstraints:
    """A host dependency lock exported for wheel installation."""

    path: Path
    sha256: str

    @classmethod
    def from_host_path(cls, raw_path: str | Path) -> ThornConstraints:
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Thorn constraints path is not a file: {path}")

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(path=path, sha256=digest)


class ThornHarborAgent(BaseInstalledAgent):
    """Harbor adapter for the current local/direct ``thorn run`` path.

    The host supplies a wheel plus its source revision. The adapter uploads the
    wheel into each task container and installs it with a pinned uv-managed
    Python rather than relying on the task image's Python environment.
    """

    SUPPORTS_ATIF = False

    def __init__(
        self,
        logs_dir: Path,
        thorn_wheel_path: str,
        thorn_constraints_path: str,
        thorn_revision: str,
        task_shell_environment: str = TaskShellEnvironment.INHERIT.value,
        action_policy: str = ThornRunActionPolicy.BASELINE.value,
        history_policy: str = ThornRunHistoryPolicy.BASELINE.value,
        validation_convergence_policy: str = (
            ThornRunValidationConvergencePolicy.BASELINE.value
        ),
        read_reuse_policy: str = ThornRunReadReusePolicy.BASELINE.value,
        prompt_trace_capture: str = ThornPromptTraceCapture.REDACTED.value,
        model_name: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._thorn_wheel = ThornWheel.from_host_path(thorn_wheel_path)
        self._remote_wheel_path = (
            _REMOTE_INSTALL_ROOT / self._thorn_wheel.path.name
        )
        self._thorn_constraints = ThornConstraints.from_host_path(
            thorn_constraints_path
        )
        self._thorn_revision = ThornRevision.parse(thorn_revision)
        self._task_shell_environment = TaskShellEnvironment.parse(
            task_shell_environment,
        )
        self._action_policy = ThornRunActionPolicy.parse(action_policy)
        self._history_policy = ThornRunHistoryPolicy.parse(history_policy)
        self._validation_convergence_policy = (
            ThornRunValidationConvergencePolicy.parse(
                validation_convergence_policy,
            )
        )
        self._read_reuse_policy = ThornRunReadReusePolicy.parse(
            read_reuse_policy,
        )
        self._prompt_trace_capture = ThornPromptTraceCapture.parse(
            prompt_trace_capture,
        )
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            version=self._thorn_revision.value,
            *args,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "thorn"

    def _provenance_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "harbor_revision": HARBOR_REVISION,
            "thorn_revision": self._thorn_revision.value,
            "thorn_wheel": {
                "filename": self._thorn_wheel.path.name,
                "sha256": self._thorn_wheel.sha256,
            },
            "thorn_constraints": {
                "filename": self._thorn_constraints.path.name,
                "sha256": self._thorn_constraints.sha256,
            },
            "uv_version": UV_VERSION,
            "python_version": PYTHON_VERSION,
            "profile": "local",
            "prompt_delivery": "direct",
            "action_policy": self._action_policy.value,
            "history_policy": self._history_policy.value,
            "validation_convergence_policy": (
                self._validation_convergence_policy.value
            ),
            "read_reuse_policy": self._read_reuse_policy.value,
            "read_reuse_telemetry_schema_version": (
                READ_REUSE_TELEMETRY_SCHEMA_VERSION
            ),
            "prompt_trace_capture": self._prompt_trace_capture.value,
            "workspace": str(_REMOTE_WORKSPACE),
            "agency": "fresh-mktemp",
            "task_shell_environment": self._task_shell_environment.value,
            "supports_atif": self.SUPPORTS_ATIF,
        }

    def _write_host_provenance(self) -> Path:
        setup_directory = self.logs_dir / "setup"
        setup_directory.mkdir(parents=True, exist_ok=True)
        provenance_path = setup_directory / _PROVENANCE_FILENAME
        provenance_path.write_text(
            json.dumps(self._provenance_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return provenance_path

    def _install_command(self) -> str:
        install_log_path = _REMOTE_AGENT_LOGS / _INSTALL_LOG_FILENAME
        install_steps = (
            f"mkdir -p {shlex.quote(str(_REMOTE_UV_DIRECTORY))}",
            "command -v curl >/dev/null 2>&1 || "
            "{ echo 'curl is required to install pinned uv' >&2; exit 127; }",
            f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | "
            f"env UV_UNMANAGED_INSTALL={shlex.quote(str(_REMOTE_UV_DIRECTORY))} sh",
            f"test \"$({shlex.quote(str(_REMOTE_UV_PATH))} --version)\" = "
            f"{shlex.quote(f'uv {UV_VERSION}')}",
            f"env UV_PYTHON_INSTALL_DIR={shlex.quote(str(_REMOTE_PYTHON_DIRECTORY))} "
            f"{shlex.quote(str(_REMOTE_UV_PATH))} python install {PYTHON_VERSION}",
            f"env UV_PYTHON_INSTALL_DIR={shlex.quote(str(_REMOTE_PYTHON_DIRECTORY))} "
            "UV_PYTHON_PREFERENCE=only-managed "
            f"{shlex.quote(str(_REMOTE_UV_PATH))} venv "
            f"--python {PYTHON_VERSION} {shlex.quote(str(_REMOTE_VENV_DIRECTORY))}",
            f"{shlex.quote(str(_REMOTE_UV_PATH))} pip install "
            f"--python {shlex.quote(str(_REMOTE_VENV_DIRECTORY / 'bin/python'))} "
            f"--constraints {shlex.quote(str(_REMOTE_CONSTRAINTS_PATH))} "
            f"{shlex.quote(str(self._remote_wheel_path))}",
            f"{shlex.quote(str(_REMOTE_VENV_DIRECTORY / 'bin/python'))} "
            "-c 'import sys; assert sys.version_info[:2] == (3, 11)'",
            f"{shlex.quote(str(_REMOTE_THORN_PATH))} run --help",
        )
        return "\n".join(
            (
                "set -e",
                "(",
                *install_steps,
                f") 2>&1 | tee {shlex.quote(str(install_log_path))}",
            )
        )

    async def install(self, environment: BaseEnvironment) -> None:
        await environment.upload_file(
            self._thorn_wheel.path,
            str(self._remote_wheel_path),
        )
        await environment.upload_file(
            self._thorn_constraints.path,
            str(_REMOTE_CONSTRAINTS_PATH),
        )
        await environment.upload_file(
            self._write_host_provenance(),
            str(_REMOTE_AGENT_LOGS / _PROVENANCE_FILENAME),
        )
        await self.exec_as_root(
            environment,
            command=self._install_command(),
            timeout_sec=900,
        )

    def _run_command(self, instruction: str) -> str:
        trace_path = _REMOTE_AGENT_LOGS / _TRACE_FILENAME
        result_path = _REMOTE_AGENT_LOGS / _RESULT_FILENAME
        output_path = _REMOTE_AGENT_LOGS / _OUTPUT_FILENAME
        raw_prompt_trace_flag = (
            "--trace-raw-prompts "
            if self._prompt_trace_capture is ThornPromptTraceCapture.RAW
            else ""
        )
        return "\n".join(
            (
                *self._task_shell_environment_steps(),
                "THORN_HARBOR_AGENCY_HOME=\"$(mktemp -d "
                "/tmp/thorn-harbor-agency.XXXXXX)\"",
                f"{shlex.quote(str(_REMOTE_THORN_PATH))} run "
                "--agent-profile local "
                f"--action-policy {shlex.quote(self._action_policy.value)} "
                f"--history-policy {shlex.quote(self._history_policy.value)} "
                "--validation-convergence-policy "
                f"{shlex.quote(self._validation_convergence_policy.value)} "
                "--read-reuse-policy "
                f"{shlex.quote(self._read_reuse_policy.value)} "
                f"--workspace {shlex.quote(str(_REMOTE_WORKSPACE))} "
                '--agency "$THORN_HARBOR_AGENCY_HOME" '
                f"--trace {shlex.quote(str(trace_path))} "
                f"{raw_prompt_trace_flag}"
                f"--result-file {shlex.quote(str(result_path))} "
                f"{shlex.quote(instruction)} 2>&1 | "
                f"tee {shlex.quote(str(output_path))}",
            )
        )

    def _task_shell_environment_steps(self) -> tuple[str, ...]:
        if self._task_shell_environment is TaskShellEnvironment.INHERIT:
            return ()
        return (
            "test -f /opt/miniconda3/bin/activate",
            ". /opt/miniconda3/bin/activate",
            "conda activate testbed",
            'test "$CONDA_DEFAULT_ENV" = "testbed"',
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Harbor scopes agent extra_env around this call. Keeping provider
        # credentials out of `command` prevents them from leaking into logs.
        await self.exec_as_agent(
            environment,
            command=self._run_command(instruction),
            cwd=str(_REMOTE_WORKSPACE),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        result_path = self.logs_dir / _RESULT_FILENAME
        if not result_path.is_file():
            return

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Could not parse Thorn result file: %s", exc)
            return
        if not isinstance(result, dict):
            self.logger.warning("Thorn result file does not contain a JSON object")
            return

        token_usage = result.get("token_usage")
        total_tokens: int | None = None
        if isinstance(token_usage, dict):
            context.n_input_tokens = self._optional_int(
                token_usage.get("prompt_tokens")
            )
            context.n_output_tokens = self._optional_int(
                token_usage.get("completion_tokens")
            )
            total_tokens = self._optional_int(token_usage.get("total_tokens"))

        context.metadata = {
            "thorn": {
                "profile": "local",
                "prompt_delivery": "direct",
                "action_policy": self._action_policy.value,
                "history_policy": self._history_policy.value,
                "validation_convergence_policy": (
                    self._validation_convergence_policy.value
                ),
                "read_reuse_policy": self._read_reuse_policy.value,
                "read_reuse_telemetry_schema_version": (
                    READ_REUSE_TELEMETRY_SCHEMA_VERSION
                ),
                "prompt_trace_capture": self._prompt_trace_capture.value,
                "task_shell_environment": self._task_shell_environment.value,
                "revision": self._thorn_revision.value,
                "result_file": _RESULT_FILENAME,
                "trace_file": _TRACE_FILENAME,
                "outcome": result.get("outcome"),
                "duration_s": result.get("duration_s"),
                "error": result.get("error"),
                "total_tokens": total_tokens,
            }
        }

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value
