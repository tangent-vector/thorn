from __future__ import annotations

import json
from typing import Any

from thorn.gateway._config import ForgeSpec, GatewayConfig
from thorn.gateway._peer import PeerAccount, PeerSpec
from thorn.gateway._peer_resolution import (
    apply_peer_account_resolutions,
    resolve_peer_account_handles,
)
from thorn.tools.forge import ForgeUserIdentity


class _FakeForgeClient:
    def __init__(self, users_by_handle: dict[str, ForgeUserIdentity]) -> None:
        self._users_by_handle = users_by_handle

    def resolve_account_handle(self, handle: str) -> ForgeUserIdentity:
        user = self._users_by_handle.get(handle)
        if user is None:
            raise LookupError(f"no such user: {handle}")
        return user


def test_resolves_github_and_gitlab_peer_handles() -> None:
    config = GatewayConfig(
        forges=[
            ForgeSpec(url="https://github.com"),
            ForgeSpec(url="https://gitlab.com"),
        ],
        peers=[
            PeerSpec(
                id="ada",
                accounts=[PeerAccount(service="github", account_id="ada")],
            ),
            PeerSpec(
                id="linus",
                accounts=[PeerAccount(service="gitlab", account_id="linus")],
            ),
        ],
    )

    result = resolve_peer_account_handles(
        config,
        forge_clients_by_service={
            "github": _FakeForgeClient({
                "ada": ForgeUserIdentity(
                    account_id="1001",
                    display_handle="ada",
                    display_name="Ada",
                ),
            }),
            "gitlab": _FakeForgeClient({
                "linus": ForgeUserIdentity(
                    account_id="2002",
                    display_handle="linus",
                    display_name="Linus",
                ),
            }),
        },
        forge_types_by_service={"github": "github", "gitlab": "gitlab"},
    )

    assert result.errors == []
    assert [(r.location.peer_id, r.immutable_account_id) for r in result.resolutions] == [
        ("ada", "1001"),
        ("linus", "2002"),
    ]

    raw_config: dict[str, Any] = {
        "peers": [
            {"id": "ada", "accounts": [{"service": "github", "account_id": "ada"}]},
            {
                "id": "linus",
                "accounts": [{"service": "gitlab", "account_id": "linus"}],
            },
        ],
    }
    updated = apply_peer_account_resolutions(raw_config, result.resolutions)

    assert updated["peers"][0]["accounts"][0] == {
        "service": "github",
        "account_id": "1001",
        "display_handle": "ada",
    }
    assert updated["peers"][1]["accounts"][0] == {
        "service": "gitlab",
        "account_id": "2002",
        "display_handle": "linus",
    }
    json.dumps(updated)


def test_unresolved_handle_reports_error_without_rewrite() -> None:
    config = GatewayConfig(
        forges=[ForgeSpec(url="https://github.com")],
        peers=[
            PeerSpec(
                id="ada",
                accounts=[PeerAccount(service="github", account_id="ada")],
            ),
        ],
    )

    result = resolve_peer_account_handles(
        config,
        forge_clients_by_service={"github": _FakeForgeClient({})},
        forge_types_by_service={"github": "github"},
    )

    assert result.resolutions == []
    assert len(result.errors) == 1
    assert result.errors[0].location.peer_id == "ada"
    assert "no such user" in result.errors[0].reason


def test_already_immutable_peer_account_is_not_rewritten() -> None:
    config = GatewayConfig(
        forges=[ForgeSpec(url="https://github.com")],
        peers=[
            PeerSpec(
                id="ada",
                accounts=[
                    PeerAccount(
                        service="github",
                        account_id="1001",
                        display_handle="ada",
                    ),
                ],
            ),
        ],
    )

    result = resolve_peer_account_handles(
        config,
        forge_clients_by_service={"github": _FakeForgeClient({})},
        forge_types_by_service={"github": "github"},
    )

    assert result.resolutions == []
    assert result.errors == []
