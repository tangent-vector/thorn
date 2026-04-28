"""GitHub API connection settings (PAT vs GitHub App installation auth).

Shared by :class:`~thorn.tools.github.GitHubClient`, forge services, and the
GitHub notifications event source.  ``PyGithub`` performs JWT signing and
installation access token refresh when using app credentials.

These models are populated from JSON config (gateway forge entries plus
agent identity credentials) at runtime.  The discriminator-tagged
``GitHubAuthUnion`` lets the agent identity JSON declare either PAT or
GitHub App auth uniformly.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

from thorn.core._credentials import ServiceCredential

_DEFAULT_API_BASE = "https://api.github.com"


class GitHubPatAuth(BaseModel):
    """Authenticate with a personal access token or other static bearer token."""

    kind: Literal["pat"] = "pat"
    token: ServiceCredential = Field(
        description="PAT or OAuth access token string",
    )


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
    private_key_pem: ServiceCredential = Field(
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
