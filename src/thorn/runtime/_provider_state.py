"""Shared session metadata keys for provider availability state."""

from __future__ import annotations

PROVIDER_UNAVAILABLE_METADATA_KEY = "provider_unavailable"
"""Session metadata key used while a session waits on provider recovery."""


__all__ = ["PROVIDER_UNAVAILABLE_METADATA_KEY"]
