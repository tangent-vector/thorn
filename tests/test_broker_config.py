"""Tests for the ``BrokerConfig`` block of ``gateway.json``.

Covers the on-disk shape of both modes (``bundled`` and
``external``) and the omission-vs-explicit distinction at the
:class:`GatewayConfig` level (omission auto-fills a bundled-mode
default when sandbox is container-backed).

The admin API key is referenced via the operator-chosen environment
variable name (``admin_api_key_env_var``), never carried inline in
``gateway.json``.  Resolution from ``os.environ`` is the broker
client's responsibility (covered in ``tests/test_broker.py``);
bundled-supervisor behavior lives in
``tests/test_bundled_broker.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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
        "admin_api_key_env_var": "ONECLI_ADMIN_KEY",
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
        assert cfg.admin_api_key_env_var is None
        assert cfg.proxy_url == ""

    def test_bundled_with_admin_url_rejected(self):
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate({
                "mode": "bundled",
                "admin_url": "http://onecli-web:10254",
            })
        assert "admin_url" in str(exc.value)

    def test_bundled_with_admin_api_key_env_var_rejected(self):
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate({
                "mode": "bundled",
                "admin_api_key_env_var": "ONECLI_ADMIN_KEY",
            })
        assert "admin_api_key_env_var" in str(exc.value)

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
        assert cfg.admin_api_key_env_var == "ONECLI_ADMIN_KEY"

    def test_explicit_disabled(self):
        cfg = BrokerConfig.model_validate(
            _make_external_broker_dict(enabled=False),
        )
        assert cfg.enabled is False

    def test_admin_api_key_env_var_just_names_var_not_value(self):
        # The schema deliberately does not accept a literal ``admin_api_key``
        # value -- only the env var name is allowed in gateway.json so the
        # serialized config never carries the secret value at rest.
        cfg = BrokerConfig.model_validate(
            _make_external_broker_dict(admin_api_key_env_var="MY_OC_KEY"),
        )
        assert cfg.admin_api_key_env_var == "MY_OC_KEY"
        text = repr(cfg)
        # The env var name is OK to appear, but no secret value lives in
        # the model so there is nothing to leak.
        assert "MY_OC_KEY" in text

    def test_external_missing_admin_url_raises(self):
        bad = _make_external_broker_dict()
        del bad["admin_url"]
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate(bad)
        assert "admin_url" in str(exc.value)

    def test_external_missing_admin_api_key_env_var_raises(self):
        bad = _make_external_broker_dict()
        del bad["admin_api_key_env_var"]
        with pytest.raises(ValidationError) as exc:
            BrokerConfig.model_validate(bad)
        assert "admin_api_key_env_var" in str(exc.value)

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
        config = GatewayConfig.model_validate({})
        assert config.sandbox is not None
        assert config.sandbox.backend == "container"
        assert config.broker is not None
        assert config.broker.mode == "bundled"
        assert config.broker.enabled is True

    def test_omitted_broker_yields_none_when_sandbox_subprocess(self):
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
        assert config.broker.admin_api_key_env_var == "ONECLI_ADMIN_KEY"


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
# load_gateway_config (file-based)
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
        assert config.broker.admin_api_key_env_var == "ONECLI_ADMIN_KEY"

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


# Smoke: sandbox/broker fixtures used elsewhere should still be
# constructible by tests that hand-craft them.
class TestModelBuildSmoke:
    def test_sandbox_config_default(self):
        cfg = SandboxConfig()
        assert cfg.backend == "container"
