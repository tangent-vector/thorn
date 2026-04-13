"""GitHub API connection settings (PAT vs GitHub App installation auth).

Shared by :class:`~thorn.tools.github.GitHubClient`, forge services, and the
GitHub notifications event source.  ``PyGithub`` performs JWT signing and
installation access token refresh when using app credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

_DEFAULT_API_BASE = "https://api.github.com"


class GitHubPatAuth(BaseModel):
    """Authenticate with a personal access token or other static bearer token."""

    kind: Literal["pat"] = "pat"
    token: str = Field(description="PAT or OAuth access token string")


class GitHubAppAuth(BaseModel):
    """Authenticate as a GitHub App installation (JWT + installation token)."""

    kind: Literal["app"] = "app"
    app_id: int = Field(description="Numeric GitHub App ID (JWT iss claim)")
    installation_id: int = Field(
        description="Installation ID for the org or user account",
    )
    private_key_pem: str = Field(
        description="PEM-encoded RSA private key for the GitHub App",
    )


GitHubAuthUnion = Annotated[
    Union[GitHubPatAuth, GitHubAppAuth],
    Field(discriminator="kind"),
]


class GitHubConnectionConfig(BaseModel):
    """Settings needed to open a PyGithub client and raw REST calls."""

    base_url: str = Field(
        default=_DEFAULT_API_BASE,
        description="GitHub REST API base URL (GitHub.com or GitHub Enterprise)",
    )
    auth: GitHubAuthUnion

    @classmethod
    def from_env(cls) -> GitHubConnectionConfig:
        """Load from environment.

        If ``GITHUB_APP_ID`` is set, builds app auth (requires
        ``GITHUB_APP_INSTALLATION_ID`` and a private key via
        ``GITHUB_APP_PRIVATE_KEY`` or ``GITHUB_APP_PRIVATE_KEY_PATH``).

        Otherwise requires ``GITHUB_TOKEN`` (PAT or other bearer token).
        """
        base_url = os.environ.get("GITHUB_URL", _DEFAULT_API_BASE)
        app_id_raw = os.environ.get("GITHUB_APP_ID")
        if app_id_raw:
            inst_raw = os.environ.get("GITHUB_APP_INSTALLATION_ID")
            if not inst_raw:
                raise ValueError(
                    "GITHUB_APP_INSTALLATION_ID is required when GITHUB_APP_ID is set",
                )
            pem = _load_app_private_key_pem_from_env()
            return cls(
                base_url=base_url,
                auth=GitHubAppAuth(
                    app_id=int(app_id_raw),
                    installation_id=int(inst_raw),
                    private_key_pem=pem,
                ),
            )
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "Set GITHUB_TOKEN for PAT auth, or GITHUB_APP_ID plus "
                "GITHUB_APP_INSTALLATION_ID and a private key for GitHub App auth.",
            )
        return cls(
            base_url=base_url,
            auth=GitHubPatAuth(token=token),
        )


def _load_app_private_key_pem_from_env() -> str:
    path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH")
    inline = os.environ.get("GITHUB_APP_PRIVATE_KEY")
    if path and inline:
        raise ValueError(
            "Set only one of GITHUB_APP_PRIVATE_KEY_PATH or GITHUB_APP_PRIVATE_KEY",
        )
    if path:
        return Path(path).expanduser().read_text(encoding="utf-8")
    if inline:
        return inline.replace("\\n", "\n")
    raise ValueError(
        "GitHub App auth requires GITHUB_APP_PRIVATE_KEY or "
        "GITHUB_APP_PRIVATE_KEY_PATH",
    )
