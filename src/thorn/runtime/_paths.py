"""Agency directory layout: home vs. workspace paths.

``AgencyPaths`` captures the two directory roots that an agency needs:

- **home_root**: where agent state lives (identities, sessions, memory,
  journals).  In CLI mode this is ``{cwd}/.thorn/``; in gateway mode it
  is a user-chosen directory.

- **workspace_root**: where agents do work (clone repositories, edit
  files, run builds).  In CLI mode this is ``{cwd}`` itself; in gateway
  mode it is a separate directory.

The distinction matters because CLI mode nests the home directory *under*
the workspace (``{cwd}/.thorn/`` inside the project checkout), while
gateway mode keeps them fully independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thorn.runtime._session import AgentID, SessionKey


@dataclass(frozen=True)
class AgencyPaths:
    """Directory roots for an agency's home (state) and workspace (work).

    Provides helpers for deriving per-agent and per-session paths from
    the two roots.
    """

    home_root: Path
    """Root directory for agent state (identities, sessions, memory)."""

    workspace_root: Path
    """Root directory where agent sessions do their work."""

    @property
    def agents_root(self) -> Path:
        """Root directory for the ``SessionStore`` (agent identities + sessions)."""
        return self.home_root / "agents"

    def agent_home(self, agent_id: AgentID) -> Path:
        """Home directory for a specific agent (memory, journals, etc.).

        Lives under ``agents_root`` so that it is co-located with the
        agent's identity file and session directories.
        """
        return self.agents_root / str(agent_id)

    def session_workspace(
        self,
        agent_id: AgentID,
        session_key: SessionKey,
    ) -> Path:
        """Working directory for a specific agent session."""
        return self.workspace_root / str(agent_id) / str(session_key)

    @classmethod
    def for_cli(cls, cwd: Path) -> AgencyPaths:
        """Construct paths for CLI mode (``thorn run`` / ``thorn chat``).

        Home is ``{cwd}/.thorn/``, workspace is ``{cwd}`` itself.
        """
        return cls(
            home_root=cwd / ".thorn",
            workspace_root=cwd,
        )

    @classmethod
    def for_gateway(
        cls,
        agency_dir: Path,
        workspace_dir: Path,
    ) -> AgencyPaths:
        """Construct paths for gateway mode (``thorn serve``).

        Home and workspace are fully independent directories.
        """
        return cls(
            home_root=agency_dir,
            workspace_root=workspace_dir,
        )


__all__ = [
    "AgencyPaths",
]
