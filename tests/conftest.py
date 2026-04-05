"""Shared fixtures for thorn tests."""

from __future__ import annotations

import pytest

from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._provider import MockProvider


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
