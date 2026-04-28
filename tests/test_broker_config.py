"""Tests for the ``BrokerConfig`` block of ``gateway.json``.

Phase D introduces a top-level ``broker`` block to ``GatewayConfig``;
this file covers the on-disk shape, the omission-vs-disabled
distinction, and ``$ENV_VAR`` expansion of the admin API key (which
is itself a meta-credential).  Broker behavior (how the admin client
calls OneCLI) is covered separately in ``tests/test_broker.py`` once
the brain-side broker client lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from thorn.core._credentials import ServiceCredential
from thorn.gateway._config import (
    BrokerConfig,
    GatewayConfig,
    load_gateway_config,
)


def _make_broker_dict(**overrides: object) -> dict[str, object]:
    """Return a minimal-valid broker config dict, plus *overrides*."""
    base: dict[str, object] = {
        "admin_url": "http://onecli-web:10254",
        "admin_api_key": "oc_dummy_admin_token",
        "proxy_url": "http://onecli-gateway:10255",
        "ca_certificate_path": "/var/lib/onecli/ca.pem",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# BrokerConfig direct construction
# ---------------------------------------------------------------------------


class TestBrokerConfig:
    def test_minimal_required_fields(self):
        cfg = BrokerConfig.model_validate(_make_broker_dict())
        assert cfg.enabled is True
        assert cfg.admin_url == "http://onecli-web:10254"
        assert cfg.proxy_url == "http://onecli-gateway:10255"
        assert cfg.ca_certificate_path == "/var/lib/onecli/ca.pem"
        # admin_api_key wraps as a ServiceCredential (literal); the
        # broker-client will eventually swap it out, but at config-load
        # time it lives on the gateway and is tolerated by the audit
        # invariant because it is a meta-credential.
        assert isinstance(cfg.admin_api_key, ServiceCredential)
        assert cfg.admin_api_key.is_literal
        assert cfg.admin_api_key == "oc_dummy_admin_token"

    def test_explicit_disabled(self):
        cfg = BrokerConfig.model_validate(_make_broker_dict(enabled=False))
        assert cfg.enabled is False

    def test_admin_api_key_repr_does_not_leak_value(self):
        cfg = BrokerConfig.model_validate(
            _make_broker_dict(admin_api_key="oc_super_secret_admin"),
        )
        text = repr(cfg)
        assert "oc_super_secret_admin" not in text

    def test_missing_required_field_raises(self):
        bad = _make_broker_dict()
        del bad["admin_url"]
        with pytest.raises(ValidationError):
            BrokerConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# GatewayConfig.broker
# ---------------------------------------------------------------------------


class TestGatewayConfigBrokerBlock:
    def test_omitted_block_yields_none(self):
        config = GatewayConfig.model_validate({})
        assert config.broker is None

    def test_present_block_loads(self):
        config = GatewayConfig.model_validate({"broker": _make_broker_dict()})
        assert config.broker is not None
        assert config.broker.enabled is True
        assert config.broker.proxy_url == "http://onecli-gateway:10255"


# ---------------------------------------------------------------------------
# load_gateway_config (file-based, $ENV_VAR expansion)
# ---------------------------------------------------------------------------


class TestLoadGatewayConfigBrokerBlock:
    def test_load_with_broker(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({"broker": _make_broker_dict()}),
            encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert config.broker is not None
        assert config.broker.enabled
        assert config.broker.proxy_url == "http://onecli-gateway:10255"

    def test_admin_api_key_env_var_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # The admin API key is a credential, so it follows the
        # existing ``$ENV_VAR`` expansion path: storing it directly
        # in gateway.json is discouraged.
        monkeypatch.setenv("ONECLI_ADMIN_KEY", "oc_real_key_from_env")

        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({"broker": _make_broker_dict(
                admin_api_key="$ONECLI_ADMIN_KEY",
            )}),
            encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert config.broker is not None
        assert config.broker.admin_api_key == "oc_real_key_from_env"
        assert config.broker.admin_api_key.is_literal

    def test_admin_api_key_missing_env_raises(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({"broker": _make_broker_dict(
                admin_api_key="$DEFINITELY_NOT_SET_ONECLI_KEY",
            )}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="DEFINITELY_NOT_SET_ONECLI_KEY"):
            load_gateway_config(thorn_dir)
