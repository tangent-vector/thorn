"""Credential scope inspection results shared by forge clients."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CredentialScopeWarning:
    """One advisory credential-scope finding from a forge API."""

    summary: str
    detail: str = ""


@dataclass(frozen=True)
class BroadCredentialScopeWarning(CredentialScopeWarning):
    """A credential grants more authority than unattended work should need."""


@dataclass(frozen=True)
class MissingCredentialScopeWarning(CredentialScopeWarning):
    """A credential may be too narrow for Thorn's configured forge workflow."""


@dataclass(frozen=True)
class CredentialScopeInspection:
    """Scopes observed for a credential, plus non-fatal warnings."""

    observed_scopes: tuple[str, ...] = ()
    warnings: tuple[CredentialScopeWarning, ...] = ()


__all__ = [
    "BroadCredentialScopeWarning",
    "CredentialScopeInspection",
    "CredentialScopeWarning",
    "MissingCredentialScopeWarning",
]
