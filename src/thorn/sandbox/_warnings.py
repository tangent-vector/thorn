"""Operator warnings for sandbox configurations with security caveats."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thorn.gateway._config import AgentSandboxOverride, SandboxConfig


class SubprocessSandboxWarningSurface(StrEnum):
    """User-facing surface that is about to run subprocess-backed tools."""

    CLI_COMMAND = "cli_command"
    GATEWAY = "gateway"


@dataclass(frozen=True)
class SubprocessSandboxWarning:
    """Rendered warning for host-subprocess sandbox execution."""

    surface: SubprocessSandboxWarningSurface

    def lines(self) -> tuple[str, ...]:
        if self.surface is SubprocessSandboxWarningSurface.CLI_COMMAND:
            return (
                "Security warning: this CLI command enables shell-capable "
                "agents using the host subprocess sandbox.",
                "Tools such as run_shell execute on this host with your "
                "user privileges; container isolation is not configured "
                "for thorn run/chat.",
                "Use subprocess sandboxing only for local/dev work you "
                "trust. Do not use it for hosted or shared deployments.",
            )
        return (
            "Security warning: sandbox.backend is subprocess.",
            "Shell-capable tools such as run_shell execute on the gateway "
            "host with the gateway user's privileges.",
            "Use the container backend for hosted or shared deployments; "
            "subprocess mode is only suitable for local/dev opt-in use.",
        )

    def plain_text(self) -> str:
        """Render a log-safe single message with all warning lines."""
        return " ".join(self.lines())

    def rich_text(self) -> str:
        """Render the warning for Thorn's Rich-powered CLI console."""
        return "\n".join(self.lines())


def uses_subprocess_sandbox_backend(
    agency: "SandboxConfig | None",
    override: "AgentSandboxOverride | None" = None,
) -> bool:
    """Return whether the effective sandbox backend is subprocess."""
    from thorn.sandbox._resolve import resolve_sandbox_config

    return resolve_sandbox_config(agency, override).backend == "subprocess"


__all__ = [
    "SubprocessSandboxWarning",
    "SubprocessSandboxWarningSurface",
    "uses_subprocess_sandbox_backend",
]
