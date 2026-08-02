"""Resolve peer account handles to immutable forge account IDs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

from thorn.gateway._config import (
    _FORGE_TYPES_WITH_IMMUTABLE_USER_IDS,
    GatewayConfig,
)
from thorn.tools.forge import ForgeUserIdentity


class PeerAccountHandleResolver(Protocol):
    """Forge-client surface needed by ``thorn serve resolve-peers``."""

    def resolve_account_handle(self, handle: str) -> ForgeUserIdentity:
        """Resolve a user-facing handle to an immutable platform account ID."""
        ...


@dataclass(frozen=True)
class PeerAccountLocation:
    """Index and labels for one ``peers[].accounts[]`` entry."""

    peer_index: int
    account_index: int
    peer_id: str
    service_name: str
    original_account_id: str


@dataclass(frozen=True)
class PeerAccountResolution:
    """Successful rewrite for one peer account."""

    location: PeerAccountLocation
    immutable_account_id: str
    display_handle: str


@dataclass(frozen=True)
class PeerAccountResolutionError:
    """Failed rewrite for one peer account."""

    location: PeerAccountLocation
    reason: str


@dataclass(frozen=True)
class PeerAccountResolutionResult:
    """All rewrites and failures found during one resolver run."""

    resolutions: list[PeerAccountResolution]
    errors: list[PeerAccountResolutionError]


def resolve_peer_account_handles(
    config: GatewayConfig,
    *,
    forge_clients_by_service: dict[str, PeerAccountHandleResolver],
    forge_types_by_service: dict[str, str],
) -> PeerAccountResolutionResult:
    """Resolve handle-only GitHub/GitLab peer accounts in *config*.

    Accounts already using a numeric immutable ID are left alone.
    Services whose forge type is not known to expose immutable user
    IDs are left alone as well.
    """
    resolutions: list[PeerAccountResolution] = []
    errors: list[PeerAccountResolutionError] = []

    for peer_index, peer in enumerate(config.peers):
        for account_index, account in enumerate(peer.accounts):
            forge_type = forge_types_by_service.get(account.service, "")
            if forge_type not in _FORGE_TYPES_WITH_IMMUTABLE_USER_IDS:
                continue
            if account.account_id.isdigit():
                continue

            location = PeerAccountLocation(
                peer_index=peer_index,
                account_index=account_index,
                peer_id=peer.id,
                service_name=account.service,
                original_account_id=account.account_id,
            )
            client = forge_clients_by_service.get(account.service)
            if client is None:
                errors.append(PeerAccountResolutionError(
                    location=location,
                    reason=(
                        "no authenticated agent account is available "
                        f"for service {account.service!r}"
                    ),
                ))
                continue

            try:
                resolved = client.resolve_account_handle(account.account_id)
            except Exception as exc:
                errors.append(PeerAccountResolutionError(
                    location=location,
                    reason=str(exc),
                ))
                continue

            if not resolved.account_id.isdigit():
                errors.append(PeerAccountResolutionError(
                    location=location,
                    reason=(
                        "forge returned a non-numeric account ID "
                        f"{resolved.account_id!r}"
                    ),
                ))
                continue

            resolutions.append(PeerAccountResolution(
                location=location,
                immutable_account_id=resolved.account_id,
                display_handle=(
                    resolved.display_handle or account.account_id
                ),
            ))

    return PeerAccountResolutionResult(
        resolutions=resolutions,
        errors=errors,
    )


def apply_peer_account_resolutions(
    raw_config: dict[str, Any],
    resolutions: list[PeerAccountResolution],
) -> dict[str, Any]:
    """Return a raw ``gateway.json`` dict with peer-account rewrites applied."""
    updated = copy.deepcopy(raw_config)
    peers = updated.setdefault("peers", [])
    if not isinstance(peers, list):
        raise ValueError("gateway.json field `peers` must be a list")

    for resolution in resolutions:
        location = resolution.location
        try:
            peer_entry = peers[location.peer_index]
            accounts = peer_entry["accounts"]
            account_entry = accounts[location.account_index]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                "gateway.json changed while resolving peer accounts; "
                "reload it and run `thorn serve resolve-peers` again"
            ) from exc
        if not isinstance(account_entry, dict):
            raise ValueError(
                "gateway.json peer account entries must be objects"
            )
        account_entry["account_id"] = resolution.immutable_account_id
        account_entry["display_handle"] = resolution.display_handle

    return updated


__all__ = [
    "PeerAccountHandleResolver",
    "PeerAccountLocation",
    "PeerAccountResolution",
    "PeerAccountResolutionError",
    "PeerAccountResolutionResult",
    "apply_peer_account_resolutions",
    "resolve_peer_account_handles",
]
