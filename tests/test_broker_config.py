"""Tests for the ``BrokerConfig`` block of ``gateway.json``.

Covers the on-disk shape of both modes (``bundled`` and
``external``), the omission-vs-explicit distinction at the
:class:`GatewayConfig` level (omission auto-fills a bundled-mode
default when sandbox is container-backed), and ``$ENV_VAR``
expansion of the admin API key.  Broker behavior (how the admin
client calls OneCLI) is covered separately in
``tests/test_broker.py``; bundled-supervisor behavior lives in
``tests/test_bundled_broker.py``.
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
    SandboxConfig,
    load_gateway_config,
)


def _make_external_broker_dict(**overrides: object) -> dict[str, object]:
    """Return a minimal-valid ``mode='external'`` broker config dict."""
    base: dict[str, object] = {
        "mode": "external",
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


class TestBrokerConfigBundledMode:
    def test_default_mode_is_bundled(self):
        cfg = BrokerConfig()
        assert cfg.mode == "bundled"
        assert cfg.enabled is True
        assert cfg.admin_url == ""
        assert cfg.admin_api_key is None
        assert cfg.proxy_url == ""

    def test_bundled_with_admin_url_rejected(self):
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate({
                "mode": "bundled",
                "admin_url": "http://onecli-web:10254",
            })
        # The validator names the stray field so the operator knows
        # what to remove (or that they meant ``mode='external'``).
        assert "admin_url" in str(exc.value)

    def test_bundled_with_admin_api_key_rejected(self):
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate({
                "mode": "bundled",
                "admin_api_key": "oc_explicit",
            })
        assert "admin_api_key" in str(exc.value)

    def test_bundled_with_proxy_url_rejected(self):
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate({
                "mode": "bundled",
                "proxy_url": "http://onecli-gateway:10255",
            })
        assert "proxy_url" in str(exc.value)

    def test_bundled_disabled_loads(self):
        cfg = BrokerConfig.model_validate({
            "mode": "bundled", "enabled": False,
        })
        assert cfg.mode == "bundled"
        assert cfg.enabled is False


class TestBrokerConfigExternalMode:
    def test_minimal_required_fields(self):
        cfg = BrokerConfig.model_validate(_make_external_broker_dict())
        assert cfg.mode == "external"
        assert cfg.enabled is True
        assert cfg.admin_url == "http://onecli-web:10254"
        assert cfg.proxy_url == "http://onecli-gateway:10255"
        assert cfg.ca_certificate_path == "/var/lib/onecli/ca.pem"
        assert isinstance(cfg.admin_api_key, ServiceCredential)
        assert cfg.admin_api_key.is_literal
        assert cfg.admin_api_key == "oc_dummy_admin_token"

    def test_explicit_disabled(self):
        cfg = BrokerConfig.model_validate(_make_external_broker_dict(enabled=False))
        assert cfg.enabled is False

    def test_admin_api_key_repr_does_not_leak_value(self):
        cfg = BrokerConfig.model_validate(
            _make_external_broker_dict(admin_api_key="oc_super_secret_admin"),
        )
        text = repr(cfg)
        assert "oc_super_secret_admin" not in text

    def test_external_missing_admin_url_raises(self):
        bad = _make_external_broker_dict()
        del bad["admin_url"]
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate(bad)
        assert "admin_url" in str(exc.value)

    def test_external_missing_admin_api_key_raises(self):
        bad = _make_external_broker_dict()
        del bad["admin_api_key"]
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate(bad)
        assert "admin_api_key" in str(exc.value)

    def test_external_missing_proxy_url_raises(self):
        bad = _make_external_broker_dict()
        del bad["proxy_url"]
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate(bad)
        assert "proxy_url" in str(exc.value)


# ---------------------------------------------------------------------------
# GatewayConfig.broker (auto-fill rules)
# ---------------------------------------------------------------------------


class TestGatewayConfigBrokerBlock:
    def test_omitted_yields_bundled_default_when_sandbox_container(self):
        # Empty config: GatewayConfig defaults sandbox to container,
        # then auto-fills broker as a bundled-mode default since the
        # resolved sandbox backend is container.
        config = GatewayConfig.model_validate({})
        assert config.sandbox is not None
        assert config.sandbox.backend == "container"
        assert config.broker is not None
        assert config.broker.mode == "bundled"
        assert config.broker.enabled is True

    def test_omitted_broker_yields_none_when_sandbox_subprocess(self):
        # Operator explicitly opts out of the container backend; the
        # broker auto-fill is suppressed because a bundled broker
        # without a container has nothing to inject the proxy into.
        config = GatewayConfig.model_validate({
            "sandbox": {"backend": "subprocess"},
        })
        assert config.sandbox is not None
        assert config.sandbox.backend == "subprocess"
        assert config.broker is None

    def test_explicit_broker_disabled_preserved(self):
        config = GatewayConfig.model_validate({
            "broker": {"mode": "bundled", "enabled": False},
        })
        assert config.broker is not None
        assert config.broker.enabled is False

    def test_explicit_external_broker_loads(self):
        config = GatewayConfig.model_validate(
            {"broker": _make_external_broker_dict()},
        )
        assert config.broker is not None
        assert config.broker.mode == "external"
        assert config.broker.proxy_url == "http://onecli-gateway:10255"


class TestGatewayConfigSandboxDefault:
    def test_omitted_sandbox_defaults_to_container_backend(self):
        config = GatewayConfig.model_validate({})
        assert config.sandbox is not None
        assert config.sandbox.backend == "container"

    def test_explicit_empty_sandbox_picks_container(self):
        config = GatewayConfig.model_validate({"sandbox": {}})
        assert config.sandbox is not None
        assert config.sandbox.backend == "container"

    def test_explicit_subprocess_sandbox_preserved(self):
        config = GatewayConfig.model_validate({
            "sandbox": {"backend": "subprocess"},
        })
        assert config.sandbox is not None
        assert config.sandbox.backend == "subprocess"

    def test_runtime_default_unchanged(self):
        # Critical invariant: the runtime-level
        # _DEFAULT_AGENCY_SANDBOX (used when Runtime is constructed
        # with sandbox_config=None, e.g. by ``thorn run`` /
        # ``thorn chat``) stays as subprocess so that flipping the
        # GatewayConfig schema default does not silently sandbox the
        # CLI commands.
        from thorn.sandbox._resolve import _DEFAULT_AGENCY_SANDBOX

        assert _DEFAULT_AGENCY_SANDBOX.backend == "subprocess"


# ---------------------------------------------------------------------------
# load_gateway_config (file-based, $ENV_VAR expansion)
# ---------------------------------------------------------------------------


class TestLoadGatewayConfigBrokerBlock:
    def test_load_with_external_broker(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({"broker": _make_external_broker_dict()}),
            encoding="utf-8",
        )

        config = load_gateway_config(thorn_dir)
        assert config.broker is not None
        assert config.broker.mode == "external"
        assert config.broker.enabled
        assert config.broker.proxy_url == "http://onecli-gateway:10255"

    def test_load_empty_config_auto_fills_bundled_broker(self, tmp_path: Path):
        thorn_dir = tmp_path / ".thorn"
        thorn_dir.mkdir()
        (thorn_dir / "gateway.json").write_text(
            json.dumps({}), encoding="utf-8",
        )
        config = load_gateway_config(thorn_dir)
        assert config.sandbox is not None
        assert config.sandbox.backend == "container"
        assert config.broker is not None
        assert config.broker.mode == "bundled"

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
            json.dumps({"broker": _make_external_broker_dict(
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
            json.dumps({"broker": _make_external_broker_dict(
                admin_api_key="$DEFINITELY_NOT_SET_ONECLI_KEY",
            )}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="DEFINITELY_NOT_SET_ONECLI_KEY"):
            load_gateway_config(thorn_dir)


# Smoke: sandbox/broker fixtures used elsewhere should still be
# constructible by tests that hand-craft them.
class TestModelBuildSmoke:
    def test_sandbox_config_default(self):
        # Used as a fixture in numerous downstream tests.
        cfg = SandboxConfig()
        assert cfg.backend == "container"
