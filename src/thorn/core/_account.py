"""Agent account and credential models.

An agent's identity on an external service is represented by an
account entry under the ``"accounts"`` key of its identity JSON file.
Today only forge accounts exist (``service`` names a registered forge
host such as ``"github"`` or ``"gitlab"``); the design preserves
extension space for additional service families (``"git"``, ``"email"``,
etc.) by carrying a ``service`` discriminator on every account entry.

The on-disk shape is a single flat list::

    "accounts": [
      { "service": "github",
        "credentials": {"kind": "pat", "token": "$GITHUB_TOKEN"},
        "git_user_name": "thorn-agent",
        "git_user_email": "thorn@thorn" }
    ]

Credential types are themselves a discriminated union keyed on
``"kind"``:

- ``"pat"`` / ``"app"`` -- reuse the GitHub auth models from
  :mod:`thorn.tools._github_connection`.
- ``"gitlab-pat"`` -- a GitLab personal access token.

The :func:`resolve_forge_account` helper finds the right forge
account on an agent and is the primary entry point for code that
needs forge credentials at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import BaseModel, Field

from thorn.core._credentials import ServiceCredential
from thorn.tools._github_connection import GitHubAppAuth, GitHubPatAuth

if TYPE_CHECKING:
    from thorn.core._agent import Agent


class GitLabCredentials(BaseModel):
    """Credentials for authenticating to a GitLab forge."""

    kind: Literal["gitlab-pat"] = "gitlab-pat"
    token: ServiceCredential = Field(
        description="Personal access token with 'api' scope",
    )


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

    ``service`` references the name of a :class:`ForgeHostService`
    registered on the runtime (e.g. ``"github"``).  ``credentials``
    carry the secret material needed to authenticate API calls and
    git operations.  ``git_user_name`` and ``git_user_email`` are the
    non-secret identity used for git commits when operating on this
    forge.

    Future account-service families (``"git"``, ``"email"``, ...)
    will be sibling classes in the same ``accounts`` list,
    discriminated by the ``service`` field at parse time.
    """

    service: str = Field(
        description=(
            "Name of the service this account belongs to (e.g. the "
            "registered forge name 'github')."
        ),
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
    Today every entry is a :class:`ForgeAccountConfig`; the field is
    kept open-ended for future account-service families.
    """

    accounts: list[ForgeAccountConfig] = Field(default_factory=list)

    def forge_accounts(self) -> list[ForgeAccountConfig]:
        """Return the subset of accounts that are forge accounts.

        Currently equivalent to ``self.accounts``; provided as an
        explicit accessor so call sites that only care about forge
        accounts make their intent clear (and so the implementation
        can change once non-forge account families are added).
        """
        return [a for a in self.accounts if isinstance(a, ForgeAccountConfig)]


def resolve_forge_account(
    agent: Agent,
    forge_name: str,
) -> ForgeAccountConfig:
    """Find the agent's account for the named forge.

    Walks ``agent.accounts.accounts`` looking for a forge account
    whose ``service`` field matches *forge_name*.

    Raises :class:`KeyError` with a descriptive message when no
    matching account exists.
    """
    accounts: AgentAccountsConfig | None = getattr(agent, "accounts", None)
    if accounts is None:
        raise KeyError(
            f"Agent {agent.name!r} has no accounts configured. "
            f"Cannot resolve credentials for forge {forge_name!r}."
        )
    for acct in accounts.forge_accounts():
        if acct.service == forge_name:
            return acct
    registered = [a.service for a in accounts.forge_accounts()] or ["(none)"]
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
