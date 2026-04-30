"""Tests for ``thorn.core._account`` -- account models, lookup, validation."""

from __future__ import annotations

import pytest

from thorn.core._account import (
    AccountConfig,
    AgentAccountsConfig,
    UntypedAccountConfig,
    find_credential,
    require_credential,
    resolve_account,
    validate_agent_accounts,
)
from thorn.core._agent import Agent
from thorn.core._credentials import Credential


# ---------------------------------------------------------------------------
# UntypedAccountConfig
# ---------------------------------------------------------------------------


class TestUntypedAccountConfig:
    def test_minimal_construction(self):
        acc = UntypedAccountConfig(service="github")
        assert acc.service == "github"
        assert acc.credentials == []

    def test_preserves_extra_fields(self):
        # ``extra='allow'`` so per-service fields survive parsing.
        acc = UntypedAccountConfig.model_validate({
            "service": "github",
            "credentials": [{"kind": "pat", "env_var_name": "X"}],
            "git_user_name": "bot",
            "git_user_email": "bot@example.com",
        })
        # Extra fields are accessible via attribute access.
        assert acc.git_user_name == "bot"
        assert acc.git_user_email == "bot@example.com"

    def test_is_subclass_of_account_config(self):
        # UntypedAccountConfig is a subclass of AccountConfig so
        # downstream code can ``isinstance(x, AccountConfig)`` and
        # cover both shapes uniformly.
        assert issubclass(UntypedAccountConfig, AccountConfig)

    def test_credential_validation(self):
        acc = UntypedAccountConfig.model_validate({
            "service": "github",
            "credentials": [
                {"kind": "pat", "env_var_name": "GITHUB_TOKEN"},
                {"kind": "pat", "name": "backup", "env_var_name": "BACKUP_TOKEN"},
            ],
        })
        assert len(acc.credentials) == 2
        assert all(isinstance(c, Credential) for c in acc.credentials)

    def test_service_required(self):
        with pytest.raises(ValueError):
            UntypedAccountConfig.model_validate({"service": ""})


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def _account_with_creds(*creds: tuple[str, str, str | None]) -> AccountConfig:
    return UntypedAccountConfig(
        service="github",
        credentials=[
            Credential(kind=k, env_var_name=e, name=n)
            for k, e, n in creds
        ],
    )


class TestFindCredential:
    def test_finds_first_by_kind(self):
        acc = _account_with_creds(("pat", "A", None), ("pat", "B", "backup"))
        cred = find_credential(acc, kind="pat")
        assert cred is not None
        assert cred.env_var_name == "A"

    def test_finds_by_kind_and_name(self):
        acc = _account_with_creds(("pat", "A", None), ("pat", "B", "backup"))
        cred = find_credential(acc, kind="pat", name="backup")
        assert cred is not None
        assert cred.env_var_name == "B"

    def test_returns_none_when_no_match(self):
        acc = _account_with_creds(("pat", "A", None))
        assert find_credential(acc, kind="other") is None
        assert find_credential(acc, kind="pat", name="missing") is None


class TestRequireCredential:
    def test_returns_match(self):
        acc = _account_with_creds(("pat", "A", None))
        cred = require_credential(acc, kind="pat")
        assert cred.env_var_name == "A"

    def test_raises_with_helpful_kinds(self):
        acc = _account_with_creds(("pat", "A", None))
        with pytest.raises(KeyError, match="pat"):
            require_credential(acc, kind="other")


# ---------------------------------------------------------------------------
# resolve_account
# ---------------------------------------------------------------------------


def _agent_with_typed_account(service: str) -> Agent:
    agent = Agent(name="test")
    typed = AccountConfig(service=service, credentials=[])
    agent.accounts = AgentAccountsConfig.model_construct(accounts=[typed])
    return agent


class TestResolveAccount:
    def test_finds_typed_account(self):
        agent = _agent_with_typed_account("github")
        result = resolve_account(agent, "github")
        assert result.service == "github"

    def test_raises_for_missing_service(self):
        agent = _agent_with_typed_account("github")
        with pytest.raises(KeyError, match="no account on service"):
            resolve_account(agent, "gitlab")

    def test_raises_when_account_still_untyped(self):
        agent = Agent(name="test")
        untyped = UntypedAccountConfig(service="github")
        agent.accounts = AgentAccountsConfig.model_construct(accounts=[untyped])
        with pytest.raises(TypeError, match="UntypedAccountConfig"):
            resolve_account(agent, "github")

    def test_raises_when_no_accounts_attribute(self):
        agent = Agent(name="bare")
        with pytest.raises(KeyError, match="no accounts configured"):
            resolve_account(agent, "github")


# ---------------------------------------------------------------------------
# validate_agent_accounts
# ---------------------------------------------------------------------------


class _FakeService:
    """Minimal :class:`Service`-like object for validate_agent_accounts.

    The real :class:`Service` ABC requires
    :class:`pydantic.BaseModel`-typed ``Config``; we don't need it
    here since validate_agent_accounts only calls
    ``service.validate_account``.
    """

    def __init__(self, name: str, account_cls: type[AccountConfig]) -> None:
        self._name = name
        self._account_cls = account_cls

    @property
    def name(self) -> str:
        return self._name

    def validate_account(self, raw: UntypedAccountConfig) -> AccountConfig:
        return self._account_cls.model_validate(raw.model_dump())


class _GitHubAccountFake(AccountConfig):
    git_user_name: str = ""
    git_user_email: str = ""


class TestValidateAgentAccounts:
    def _agent_with_untyped(self, *entries: dict) -> Agent:
        agent = Agent(name="test")
        agent.accounts = AgentAccountsConfig.model_construct(
            accounts=[UntypedAccountConfig.model_validate(e) for e in entries],
        )
        return agent

    def test_replaces_untyped_with_typed(self):
        agent = self._agent_with_untyped({
            "service": "github",
            "credentials": [{"kind": "pat", "env_var_name": "T"}],
            "git_user_name": "bot", "git_user_email": "bot@x",
        })
        services = {"github": _FakeService("github", _GitHubAccountFake)}
        validate_agent_accounts(agent, services.__getitem__)
        out = agent.accounts.accounts[0]
        assert isinstance(out, _GitHubAccountFake)
        assert out.git_user_name == "bot"
        # No more UntypedAccountConfig either:
        assert not isinstance(out, UntypedAccountConfig)

    def test_raises_when_service_unknown(self):
        agent = self._agent_with_untyped({
            "service": "unknown", "credentials": [],
        })
        with pytest.raises(ValueError, match="unknown"):
            validate_agent_accounts(agent, lambda _: (_ for _ in ()).throw(KeyError()))

    def test_idempotent_on_already_typed(self):
        agent = Agent(name="test")
        typed = _GitHubAccountFake(service="github")
        agent.accounts = AgentAccountsConfig.model_construct(accounts=[typed])
        validate_agent_accounts(
            agent, {"github": _FakeService("github", _GitHubAccountFake)}.__getitem__,
        )
        assert agent.accounts.accounts[0] is typed

    def test_no_op_for_agent_without_accounts(self):
        agent = Agent(name="test")
        # No accounts attribute -- must not crash.
        validate_agent_accounts(agent, lambda _: None)  # type: ignore[arg-type, return-value]


# ---------------------------------------------------------------------------
# AgentAccountsConfig wire shape
# ---------------------------------------------------------------------------


class TestAgentAccountsConfig:
    def test_default_empty(self):
        cfg = AgentAccountsConfig()
        assert cfg.accounts == []

    def test_validates_dict_entries(self):
        cfg = AgentAccountsConfig.model_validate({
            "accounts": [
                {"service": "github", "credentials": []},
            ],
        })
        # ``accounts`` is typed as list[AccountConfig], so a dict
        # entry is validated as a base AccountConfig at parse time.
        # The deserializer used by JsonSessionSerializer constructs
        # UntypedAccountConfig instances explicitly to preserve
        # per-service ``extra`` fields; that path is exercised in
        # test_runtime.py.
        assert len(cfg.accounts) == 1
        assert cfg.accounts[0].service == "github"
