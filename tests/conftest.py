"""Shared fixtures for thorn tests."""

from __future__ import annotations

import pytest

from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._provider import MockProvider

_GITHUB_APP_ENV_KEYS = (
    "GITHUB_APP_ID",
    "GITHUB_APP_INSTALLATION_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_APP_PRIVATE_KEY_PATH",
)
_GITHUB_API_URL_KEYS = (
    "GITHUB_API_URL",
    "GITHUB_URL",
)


@pytest.fixture
def github_pat_only_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset GitHub App variables so :meth:`~thorn.tools._github_connection.GitHubConnectionConfig.from_env` uses PAT mode."""
    for key in _GITHUB_APP_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def clear_github_api_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset API base URL env vars so tests can set deterministic ``GITHUB_URL`` / ``GITHUB_API_URL``."""
    for key in _GITHUB_API_URL_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def ctx(mock_provider):
    """An ExecutionContext wired to a MockProvider, installed as the
    active context for the duration of the test."""
    context = ExecutionContext(provider=mock_provider)
    token = set_context(context)
    yield context
    reset_context(token)
