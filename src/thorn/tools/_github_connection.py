"""GitHub API connection settings (PAT vs GitHub App installation auth).

Shared by :class:`~thorn.tools.github.GitHubClient`, forge services, and the
GitHub notifications event source.  ``PyGithub`` performs JWT signing and
installation access token refresh when using app credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

_DEFAULT_API_BASE = "https://api.github.com"


def github_api_base_url_from_env() -> str:
    """Return the GitHub REST API base URL from the environment.

    Prefer ``GITHUB_API_URL`` (the REST API root, e.g.
    ``https://api.github.com`` or ``https://ghe.example.com/api/v3``).
    ``GITHUB_URL`` is still read if set, for older configs — that name
    was misleading because GitHub's website URL is not the API base.
    """
    primary = os.environ.get("GITHUB_API_URL")
    if primary is not None and str(primary).strip():
        return str(primary).strip().rstrip("/")
    legacy = os.environ.get("GITHUB_URL")
    if legacy is not None and str(legacy).strip():
        return str(legacy).strip().rstrip("/")
    return _DEFAULT_API_BASE


class GitHubPatAuth(BaseModel):
    """Authenticate with a personal access token or other static bearer token."""

    kind: Literal["pat"] = "pat"
    token: str = Field(description="PAT or OAuth access token string")


class GitHubAppAuth(BaseModel):
    """Authenticate as a GitHub App installation (JWT + installation token)."""

    kind: Literal["app"] = "app"
    app_id: str = Field(
        description=(
            "GitHub App ID (digits) or OAuth Client ID — both are valid for the "
            "JWT ``iss`` claim; see GitHub App authentication docs."
        ),
    )
    installation_id: int = Field(
        description="Installation ID for the org or user account",
    )
    private_key_pem: str = Field(
        description="PEM-encoded RSA private key for the GitHub App",
    )

    @field_validator("app_id", mode="before")
    @classmethod
    def _normalize_app_id(cls, value: object) -> str:
        """Accept numeric App ID from JSON integers or Client ID / App ID strings."""
        if isinstance(value, bool):
            raise TypeError("app_id must not be a boolean")
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                raise ValueError("app_id must not be empty")
            return s
        raise TypeError(f"app_id must be str or int, got {type(value).__name__}")


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
        ``GITHUB_APP_ID`` may be the numeric App ID or the app's Client ID
        (GitHub documents both for the JWT issuer).

        Otherwise requires ``GITHUB_TOKEN`` (PAT or other bearer token).

        API host: ``GITHUB_API_URL`` (or legacy ``GITHUB_URL``).
        """
        base_url = github_api_base_url_from_env()
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
                    app_id=app_id_raw.strip(),
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
