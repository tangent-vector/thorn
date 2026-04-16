"""Agent account and credential models for forge services.

An agent's identity on a forge is represented by a
:class:`ForgeAccountConfig`, which pairs credentials (secrets) with
non-secret identity fields like the git author name and email.

Credential types are a discriminated union keyed on ``"kind"``:

- ``"pat"`` / ``"app"`` — reuse the GitHub auth models from
  :mod:`thorn.tools._github_connection`.
- ``"gitlab-pat"`` — a GitLab personal access token.

The :func:`resolve_forge_account` helper finds the right account for a
given forge on an agent, and is the primary entry point for code that
needs credentials at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import BaseModel, Field

from thorn.tools._github_connection import GitHubAppAuth, GitHubPatAuth

if TYPE_CHECKING:
    from thorn.core._agent import Agent


class GitLabCredentials(BaseModel):
    """Credentials for authenticating to a GitLab forge."""

    kind: Literal["gitlab-pat"] = "gitlab-pat"
    token: str = Field(description="Personal access token with 'api' scope")


ForgeCredentials = Annotated[
    Union[GitHubPatAuth, GitHubAppAuth, GitLabCredentials],
    Field(discriminator="kind"),
]
"""Discriminated union of all supported forge credential types.

Discriminated on the ``kind`` field:
- ``"pat"`` -> :class:`GitHubPatAuth`
- ``"app"`` -> :class:`GitHubAppAuth`
- ``"gitlab-pat"`` -> :class:`GitLabCredentials`
"""


class ForgeAccountConfig(BaseModel):
    """An agent's identity and credentials on a specific forge.

    ``forge`` references the name of a :class:`ForgeHostService`
    registered on the runtime.  ``credentials`` carry the secret
    material needed to authenticate API calls and git operations.
    ``git_user_name`` and ``git_user_email`` are the non-secret
    identity used for git commits when operating on this forge.
    """

    forge: str = Field(
        description="Name of the forge service this account belongs to",
    )
    credentials: ForgeCredentials
    git_user_name: str = Field(
        default="",
        description="Git author/committer name when operating on this forge",
    )
    git_user_email: str = Field(
        default="",
        description="Git author/committer email when operating on this forge",
    )


class AgentAccountsConfig(BaseModel):
    """All accounts declared for an agent.

    Parsed from the ``"accounts"`` key in an agent's ``.json`` file.
    """

    forge_accounts: list[ForgeAccountConfig] = Field(default_factory=list)


def resolve_forge_account(
    agent: Agent,
    forge_name: str,
) -> ForgeAccountConfig:
    """Find the agent's account for the named forge.

    Walks ``agent.accounts.forge_accounts`` looking for an entry whose
    ``forge`` field matches *forge_name*.

    Raises :class:`KeyError` with a descriptive message when no
    matching account exists.
    """
    accounts: AgentAccountsConfig | None = getattr(agent, "accounts", None)
    if accounts is None:
        raise KeyError(
            f"Agent {agent.name!r} has no accounts configured. "
            f"Cannot resolve credentials for forge {forge_name!r}."
        )
    for acct in accounts.forge_accounts:
        if acct.forge == forge_name:
            return acct
    registered = [a.forge for a in accounts.forge_accounts] or ["(none)"]
    raise KeyError(
        f"Agent {agent.name!r} has no account on forge {forge_name!r}. "
        f"Configured forge accounts: {', '.join(registered)}"
    )


__all__ = [
    "AgentAccountsConfig",
    "ForgeAccountConfig",
    "ForgeCredentials",
    "GitLabCredentials",
    "resolve_forge_account",
]
