"""Tests for thorn.core._account -- credential models and account resolution."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from thorn.core._account import (
    AgentAccountsConfig,
    ForgeAccountConfig,
    GitLabCredentials,
    resolve_forge_account,
)
from thorn.core._agent import Agent
from thorn.tools._github_connection import GitHubAppAuth, GitHubPatAuth
from thorn.tools.forge import (
    GitHubForgeClient,
    GitHubForgeService,
    GitLabForgeClient,
    GitLabForgeService,
    GitLabForgeServiceConfig,
)


# ---------------------------------------------------------------------------
# Credential model construction
# ---------------------------------------------------------------------------


class TestGitLabCredentials:
    def test_defaults_kind(self):
        creds = GitLabCredentials(token="glpat-abc")
        assert creds.kind == "gitlab-pat"
        assert creds.token == "glpat-abc"

    def test_round_trip_json(self):
        creds = GitLabCredentials(token="glpat-abc")
        data = creds.model_dump()
        assert data == {"kind": "gitlab-pat", "token": "glpat-abc"}
        restored = GitLabCredentials.model_validate(data)
        assert restored == creds


class TestForgeAccountConfig:
    def test_github_pat_account(self):
        acct = ForgeAccountConfig(
            forge="github-com",
            credentials=GitHubPatAuth(token="ghp_test"),
            git_user_name="bot",
            git_user_email="bot@example.com",
        )
        assert acct.forge == "github-com"
        assert acct.credentials.kind == "pat"
        assert acct.git_user_name == "bot"

    def test_github_app_account(self):
        acct = ForgeAccountConfig(
            forge="github-com",
            credentials=GitHubAppAuth(
                app_id="12345",
                installation_id=67890,
                private_key_pem="-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
            ),
            git_user_name="app-bot",
            git_user_email="app-bot@example.com",
        )
        assert acct.credentials.kind == "app"
        assert acct.credentials.app_id == "12345"

    def test_gitlab_pat_account(self):
        acct = ForgeAccountConfig(
            forge="my-gitlab",
            credentials=GitLabCredentials(token="glpat-xyz"),
        )
        assert acct.forge == "my-gitlab"
        assert acct.credentials.kind == "gitlab-pat"
        assert acct.git_user_name == ""
        assert acct.git_user_email == ""

    def test_round_trip_json(self):
        acct = ForgeAccountConfig(
            forge="github-com",
            credentials=GitHubPatAuth(token="ghp_test"),
            git_user_name="bot",
            git_user_email="bot@thorn",
        )
        data = acct.model_dump()
        restored = ForgeAccountConfig.model_validate(data)
        assert restored.forge == acct.forge
        assert restored.credentials.kind == "pat"
        assert restored.credentials.token == "ghp_test"
        assert restored.git_user_name == "bot"


class TestAgentAccountsConfig:
    def test_empty_default(self):
        cfg = AgentAccountsConfig()
        assert cfg.forge_accounts == []

    def test_multiple_accounts(self):
        cfg = AgentAccountsConfig(forge_accounts=[
            ForgeAccountConfig(
                forge="github-com",
                credentials=GitHubPatAuth(token="ghp_1"),
            ),
            ForgeAccountConfig(
                forge="my-gitlab",
                credentials=GitLabCredentials(token="glpat-2"),
            ),
        ])
        assert len(cfg.forge_accounts) == 2
        assert cfg.forge_accounts[0].forge == "github-com"
        assert cfg.forge_accounts[1].forge == "my-gitlab"


# ---------------------------------------------------------------------------
# resolve_forge_account
# ---------------------------------------------------------------------------


def _make_agent_with_accounts(
    accounts: AgentAccountsConfig,
    name: str = "test-agent",
) -> Agent:
    agent = Agent(name=name)
    agent.accounts = accounts
    return agent


class TestResolveForgeAccount:
    def test_finds_matching_account(self):
        accounts = AgentAccountsConfig(forge_accounts=[
            ForgeAccountConfig(
                forge="github-com",
                credentials=GitHubPatAuth(token="ghp_abc"),
                git_user_name="bot",
                git_user_email="bot@thorn",
            ),
        ])
        agent = _make_agent_with_accounts(accounts)
        result = resolve_forge_account(agent, "github-com")
        assert result.forge == "github-com"
        assert result.credentials.token == "ghp_abc"
        assert result.git_user_name == "bot"

    def test_finds_second_account(self):
        accounts = AgentAccountsConfig(forge_accounts=[
            ForgeAccountConfig(
                forge="github-com",
                credentials=GitHubPatAuth(token="ghp_first"),
            ),
            ForgeAccountConfig(
                forge="my-gitlab",
                credentials=GitLabCredentials(token="glpat-second"),
            ),
        ])
        agent = _make_agent_with_accounts(accounts)
        result = resolve_forge_account(agent, "my-gitlab")
        assert result.credentials.kind == "gitlab-pat"

    def test_raises_for_missing_forge(self):
        accounts = AgentAccountsConfig(forge_accounts=[
            ForgeAccountConfig(
                forge="github-com",
                credentials=GitHubPatAuth(token="ghp_abc"),
            ),
        ])
        agent = _make_agent_with_accounts(accounts)
        with pytest.raises(KeyError, match="no account on forge 'other-forge'"):
            resolve_forge_account(agent, "other-forge")

    def test_raises_for_no_accounts_attribute(self):
        agent = Agent(name="bare-agent")
        with pytest.raises(KeyError, match="no accounts configured"):
            resolve_forge_account(agent, "github-com")

    def test_raises_for_empty_accounts(self):
        agent = _make_agent_with_accounts(AgentAccountsConfig())
        with pytest.raises(KeyError, match="no account on forge"):
            resolve_forge_account(agent, "github-com")


# ---------------------------------------------------------------------------
# ForgeHostService.authenticated_client / git_https_password_for
# ---------------------------------------------------------------------------


class TestGitLabForgeServiceAccountAuth:
    def _make_service(self) -> GitLabForgeService:
        config = GitLabForgeServiceConfig(
            url="https://gitlab.example.com",
            token="old-baked-in-token",
        )
        return GitLabForgeService(config, service_name="my-gitlab")

    def test_authenticated_client_returns_forge_client(self):
        svc = self._make_service()
        creds = GitLabCredentials(token="new-agent-token")
        client = svc.authenticated_client(creds)
        assert isinstance(client, GitLabForgeClient)

    def test_git_https_password_for_returns_token(self):
        svc = self._make_service()
        creds = GitLabCredentials(token="new-agent-token")
        assert svc.git_https_password_for(creds) == "new-agent-token"

    def test_rejects_github_credentials(self):
        svc = self._make_service()
        creds = GitHubPatAuth(token="ghp_wrong")
        with pytest.raises(TypeError, match="GitLabCredentials"):
            svc.authenticated_client(creds)

    def test_rejects_github_credentials_for_password(self):
        svc = self._make_service()
        creds = GitHubPatAuth(token="ghp_wrong")
        with pytest.raises(TypeError, match="GitLabCredentials"):
            svc.git_https_password_for(creds)

    def test_url_property(self):
        svc = self._make_service()
        assert svc.url == "https://gitlab.example.com"


class TestGitHubForgeServiceAccountAuth:
    """Test the account-based auth methods on GitHubForgeService.

    These tests mock out GitHubClient to avoid needing PyGithub
    installed, which mirrors how the existing test_forge.py tests work.
    """

    def _make_service(self) -> GitHubForgeService:
        from thorn.tools._github_connection import GitHubConnectionConfig, GitHubPatAuth
        config = GitHubConnectionConfig(
            base_url="https://api.github.com",
            auth=GitHubPatAuth(token="old-baked-in-token"),
        )
        return GitHubForgeService(config, service_name="github-com")

    def test_authenticated_client_with_pat(self):
        svc = self._make_service()
        creds = GitHubPatAuth(token="ghp_new")

        mock_gh_client = MagicMock()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "thorn.tools.forge.GitHubForgeClient",
                lambda client: MagicMock(spec=GitHubForgeClient),
            )
            mp.setattr(
                "thorn.tools.github.GitHubClient",
                lambda config: mock_gh_client,
            )
            client = svc.authenticated_client(creds)
            assert client is not None

    def test_rejects_gitlab_credentials(self):
        svc = self._make_service()
        creds = GitLabCredentials(token="glpat-wrong")
        with pytest.raises(TypeError, match="GitHubPatAuth or GitHubAppAuth"):
            svc.authenticated_client(creds)

    def test_rejects_gitlab_credentials_for_password(self):
        svc = self._make_service()
        creds = GitLabCredentials(token="glpat-wrong")
        with pytest.raises(TypeError, match="GitHubPatAuth or GitHubAppAuth"):
            svc.git_https_password_for(creds)

    def test_base_url_property(self):
        svc = self._make_service()
        assert svc.base_url == "https://api.github.com"
