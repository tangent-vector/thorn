"""Shared types for Thorn external-content provenance."""

from __future__ import annotations

from enum import StrEnum


class ExternalContentPeerStatus(StrEnum):
    """Peer labels attached to an ``[external-content]`` block."""

    PEER = "yes"
    NON_PEER = "no"
    UNKNOWN = "unknown"


__all__ = ["ExternalContentPeerStatus"]
