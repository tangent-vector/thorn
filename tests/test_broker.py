"""Tests for :mod:`thorn.gateway._broker` (Phase D broker client).

These tests intercept HTTP traffic with :class:`httpx.MockTransport`
so the wire shapes of every admin-API call are pinned.  R1 (proxy
URL composition) and R2 (admin endpoints, auth header, request /
response shapes) are baked into the assertions here so a future
OneCLI upstream change is a noisy test failure rather than a silent
agent-load bug.

The ``register_agent_with_broker`` orchestration tests drive the
service-driven brokering protocol (``BrokerableService`` +
``BrokerCredentialPlan``) so the broker code's only knowledge of
upstream services flows through that interface.
"""

from __future__ import annotations

import json as _json
from typing import Any

import httpx
import pytest

from thorn.core._account import (
    AccountConfig,
    AgentAccountsConfig,
    UntypedAccountConfig,
)
from thorn.core._agent import Agent
from thorn.core._brokering import (
    BrokerableService,
    BrokerCredentialPlan,
    HeaderInjection,
    ParamInjection,
)
from thorn.core._credentials import Credential, ServiceCredential
from thorn.core._service import Service
from thorn.gateway._broker import (
    AgentRegistration,
    BrokerBinding,
    BrokerClient,
    BrokerError,
    SecretRegistration,
    _broker_identifier_for_agent,
    _compose_proxy_url,
    admin_api_key_from_env,
    register_agent_with_broker,
)
from thorn.gateway._config import BrokerConfig

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _broker_config(**overrides: Any) -> BrokerConfig:
    """Build an external-mode :class:`BrokerConfig` for client tests.

    External mode because client-level tests exercise an already-
    running broker with explicit URLs (the bundled-mode supervisor
    is separately covered in ``test_bundled_broker.py``).
    """
    base: dict[str, Any] = {
        "mode": "external",
        "admin_url": "http://onecli-web:10254",
        "admin_api_key_env_var": "ONECLI_ADMIN_KEY",
        "proxy_url": "http://onecli-gateway:10255",
    }
    base.update(overrides)
    return BrokerConfig.model_validate(base)


def _client_with_router(
    config: BrokerConfig,
    handler: Any,
    *,
    admin_api_key: ServiceCredential | None = None,
) -> BrokerClient:
    return BrokerClient(
        config,
        admin_api_key=admin_api_key or ServiceCredential("oc_admin_dummy"),
        transport=httpx.MockTransport(handler),
    )


def _read_json(request: httpx.Request) -> dict[str, Any]:
    return _json.loads(request.content.decode("utf-8")) if request.content else {}


# ---------------------------------------------------------------------------
# BrokerClient.__init__ guards
# ---------------------------------------------------------------------------


class TestBrokerClientGuards:
    def test_rejects_empty_admin_url(self):
        cfg = BrokerConfig.model_validate(
            {"mode": "bundled"},
        )
        with pytest.raises(BrokerError, match="admin_url"):
            BrokerClient(cfg, admin_api_key=ServiceCredential("oc_x"))

    def test_rejects_empty_api_key(self):
        cfg = _broker_config()
        with pytest.raises(BrokerError, match="non-empty admin API key"):
            BrokerClient(cfg, admin_api_key=ServiceCredential(""))


# ---------------------------------------------------------------------------
# Auth header (R2)
# ---------------------------------------------------------------------------


class TestAuthHeader:
    def test_admin_calls_send_bearer_oc_token(self):
        captured: dict[str, str | None] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers.get("Authorization")
            return httpx.Response(200, json=[])

        client = _client_with_router(
            _broker_config(), handler,
            admin_api_key=ServiceCredential("oc_real_key_xyz"),
        )
        try:
            client.fetch_ca_certificate()
        finally:
            client.close()

        assert captured["authorization"] == "Bearer oc_real_key_xyz"


# ---------------------------------------------------------------------------
# Agent registration (R2)
# ---------------------------------------------------------------------------


class TestAgentRegistration:
    def test_create_then_regenerate(self):
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, request.url.path))
            if request.method == "POST" and request.url.path == "/api/agents":
                body = _read_json(request)
                assert body == {"identifier": "thornbot", "name": "Thorn Bot"}
                return httpx.Response(
                    201,
                    json={
                        "id": "agent-123",
                        "name": "Thorn Bot",
                        "identifier": "thornbot",
                        "createdAt": "2026-04-28T00:00:00Z",
                    },
                )
            if (
                request.method == "POST"
                and request.url.path == "/api/agents/agent-123/regenerate-token"
            ):
                return httpx.Response(
                    200, json={"accessToken": "aoc_fresh_token_xyz"},
                )
            return httpx.Response(404)

        with _client_with_router(_broker_config(), handler) as broker:
            registration = broker.register_agent(
                identifier="thornbot", name="Thorn Bot",
            )

        assert isinstance(registration, AgentRegistration)
        assert registration.agent_id == "agent-123"
        assert isinstance(registration.access_token, ServiceCredential)
        assert registration.access_token == "aoc_fresh_token_xyz"
        assert calls == [
            ("POST", "/api/agents"),
            ("POST", "/api/agents/agent-123/regenerate-token"),
        ]

    def test_create_failure_raises_broker_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(409, json={"error": "identifier exists"})

        with _client_with_router(_broker_config(), handler) as broker:
            with pytest.raises(BrokerError, match="HTTP 409"):
                broker.register_agent(identifier="dup", name="dup")

    def test_unexpected_create_response_shape_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={"unexpected": "shape"})

        with _client_with_router(_broker_config(), handler) as broker:
            with pytest.raises(BrokerError, match="unexpected response shape"):
                broker.register_agent(identifier="x", name="x")

    def test_delete_agent_treats_404_as_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "agent not found"})

        with _client_with_router(_broker_config(), handler) as broker:
            broker.delete_agent("agent-already-gone")  # no exception

    def test_delete_agent_propagates_other_errors(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal"})

        with _client_with_router(_broker_config(), handler) as broker:
            with pytest.raises(BrokerError, match="HTTP 500"):
                broker.delete_agent("agent-bad")


# ---------------------------------------------------------------------------
# Secret registration (R2)
# ---------------------------------------------------------------------------


class TestSecretRegistration:
    def test_header_injection_wire_shape(self):
        captured_body: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == "/api/secrets"
            captured_body.update(_read_json(request))
            return httpx.Response(201, json={"id": "secret-1"})

        with _client_with_router(_broker_config(), handler) as broker:
            result = broker.register_secret(
                name="ghp", value="ghp_real_pat",
                host_pattern="api.github.com", path_pattern="/*",
                injection=HeaderInjection(
                    header_name="Authorization",
                    value_format="Bearer {value}",
                ),
            )

        assert isinstance(result, SecretRegistration)
        assert result.secret_id == "secret-1"
        assert captured_body == {
            "name": "ghp",
            "type": "generic",
            "value": "ghp_real_pat",
            "hostPattern": "api.github.com",
            "pathPattern": "/*",
            "injectionConfig": {
                "headerName": "Authorization",
                "valueFormat": "Bearer {value}",
            },
        }

    def test_param_injection_wire_shape(self):
        captured_body: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(_read_json(request))
            return httpx.Response(201, json={"id": "secret-2"})

        with _client_with_router(_broker_config(), handler) as broker:
            broker.register_secret(
                name="api-key", value="raw-string",
                host_pattern="api.example.com",
                injection=ParamInjection(param_name="apikey"),
            )

        assert captured_body["injectionConfig"] == {
            "paramName": "apikey",
            "paramFormat": "{value}",
        }
        # path_pattern omitted -> not in wire body.
        assert "pathPattern" not in captured_body

    def test_register_secret_failure_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad"})

        with _client_with_router(_broker_config(), handler) as broker:
            with pytest.raises(BrokerError, match="HTTP 400"):
                broker.register_secret(
                    name="x", value="v",
                    host_pattern="example.com",
                    injection=HeaderInjection(header_name="X-Auth"),
                )


# ---------------------------------------------------------------------------
# Bindings (R2)
# ---------------------------------------------------------------------------


class TestBindings:
    def test_bind_secrets_replaces_via_put(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["body"] = _read_json(request)
            return httpx.Response(200, json={})

        with _client_with_router(_broker_config(), handler) as broker:
            broker.bind_secrets_to_agent(
                "agent-1", ["secret-a", "secret-b", "secret-c"],
            )

        assert captured["method"] == "PUT"
        assert captured["path"] == "/api/agents/agent-1/secrets"
        assert captured["body"] == {
            "secretIds": ["secret-a", "secret-b", "secret-c"],
        }


# ---------------------------------------------------------------------------
# Proxy URL composition (R1)
# ---------------------------------------------------------------------------


class TestProxyUrlComposition:
    def test_basic_form(self):
        url = _compose_proxy_url("http://broker:10255", "aoc_simple_token")
        assert url == "http://x:aoc_simple_token@broker:10255"

    def test_replaces_existing_userinfo(self):
        url = _compose_proxy_url("http://stale:creds@broker:10255", "aoc_x")
        assert url == "http://x:aoc_x@broker:10255"

    def test_percent_encodes_token(self):
        url = _compose_proxy_url("http://broker:10255", "aoc_a/b@c")
        assert url == "http://x:aoc_a%2Fb%40c@broker:10255"

    def test_rejects_non_url(self):
        with pytest.raises(ValueError, match="not a full URL"):
            _compose_proxy_url("just-a-string", "aoc_t")

    def test_proxy_url_for_agent_uses_config(self):
        with _client_with_router(
            _broker_config(proxy_url="http://onecli:10255"),
            lambda r: httpx.Response(200),
        ) as broker:
            url = broker.proxy_url_for_agent(ServiceCredential("aoc_xyz"))
        assert url == "http://x:aoc_xyz@onecli:10255"


# ---------------------------------------------------------------------------
# Identifier sanitization
# ---------------------------------------------------------------------------


class TestBrokerIdentifier:
    def test_lowercases_and_keeps_hyphens(self):
        agent = Agent(name="x", id="thorn-Bot-42")
        assert _broker_identifier_for_agent(agent) == "thorn-bot-42"

    def test_replaces_invalid_chars(self):
        agent = Agent(name="x", id="my_agent.id")
        assert _broker_identifier_for_agent(agent) == "my-agent-id"

    def test_prefixes_when_starts_with_digit(self):
        agent = Agent(name="x", id="123agent")
        result = _broker_identifier_for_agent(agent)
        assert result.startswith("a")
        assert result == "a123agent"

    def test_truncates_to_50_chars(self):
        agent = Agent(name="x", id="a" * 100)
        assert len(_broker_identifier_for_agent(agent)) == 50


# ---------------------------------------------------------------------------
# admin_api_key_from_env
# ---------------------------------------------------------------------------


class TestAdminApiKeyFromEnv:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_ADMIN_KEY", "oc_value")
        cfg = _broker_config(admin_api_key_env_var="MY_ADMIN_KEY")
        value = admin_api_key_from_env(cfg)
        assert isinstance(value, ServiceCredential)
        assert value == "oc_value"

    def test_raises_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("MY_ADMIN_KEY", raising=False)
        cfg = _broker_config(admin_api_key_env_var="MY_ADMIN_KEY")
        with pytest.raises(BrokerError, match="MY_ADMIN_KEY"):
            admin_api_key_from_env(cfg)

    def test_raises_when_field_unset(self):
        cfg = BrokerConfig.model_construct(
            mode="external",
            enabled=True,
            admin_url="http://x",
            admin_api_key_env_var=None,
            proxy_url="http://y",
            ca_certificate_path=None,
        )
        with pytest.raises(BrokerError, match="admin_api_key_env_var is unset"):
            admin_api_key_from_env(cfg)


# ---------------------------------------------------------------------------
# Service-driven registration (BrokerableService protocol)
# ---------------------------------------------------------------------------


class _FakeAccountConfig(AccountConfig):
    pass


class _FakeBrokerableService(BrokerableService):
    """Test double that returns broker plans for ``"pat"`` credentials.

    Mimics the shape of a real forge service: declares an
    :attr:`AccountConfig`, implements
    :meth:`broker_credential_plans` to emit one
    :class:`BrokerCredentialPlan` per ``"pat"`` credential it finds
    on the account, with a stable host/path pattern derived from
    the constructor args.
    """

    AccountConfig = _FakeAccountConfig

    class Config:  # minimal stand-in to satisfy Service.Config
        pass

    def __init__(
        self,
        *,
        service_name: str,
        host: str,
        path: str | None = "/*",
    ) -> None:
        self._service_name = service_name
        self._host = host
        self._path = path

    @property
    def name(self) -> str:
        return self._service_name

    def broker_credential_plans(
        self,
        account: AccountConfig,
    ) -> list[BrokerCredentialPlan]:
        plans: list[BrokerCredentialPlan] = []
        for cred in account.credentials:
            if cred.kind != "pat":
                continue
            plans.append(BrokerCredentialPlan(
                env_var_name=cred.env_var_name,
                host_pattern=self._host,
                path_pattern=self._path,
                injection=HeaderInjection(
                    header_name="Authorization",
                    value_format="Bearer {value}",
                ),
                secret_name_suffix="fake-pat",
            ))
        return plans


class _NonBrokerableService(Service):
    """A service that doesn't implement BrokerableService.

    Used to verify that accounts on such services are silently
    skipped by registration -- agents can carry accounts on
    project services / future non-credential service families
    without the broker having to know about them.
    """

    class Config:
        pass

    AccountConfig = _FakeAccountConfig

    def __init__(self, *, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class _RegistrationHandler:
    """Stateful httpx handler that emulates OneCLI's admin endpoints
    well enough for orchestration tests."""

    def __init__(self) -> None:
        self.agents: dict[str, dict[str, Any]] = {}
        self.secrets: dict[str, dict[str, Any]] = {}
        self.bindings: dict[str, list[str]] = {}
        self._next_id = 0

    def _next(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}-{self._next_id}"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        body = _read_json(request)

        if method == "POST" and path == "/api/agents":
            agent_id = self._next("agent")
            self.agents[agent_id] = {**body, "id": agent_id}
            return httpx.Response(201, json=self.agents[agent_id])

        if method == "POST" and path.startswith("/api/agents/") \
                and path.endswith("/regenerate-token"):
            return httpx.Response(200, json={"accessToken": "aoc_test_token"})

        if method == "PUT" and path.endswith("/secrets") \
                and "/api/agents/" in path:
            agent_id = path.removeprefix("/api/agents/").removesuffix("/secrets")
            self.bindings[agent_id] = list(body["secretIds"])
            return httpx.Response(200, json={})

        if method == "POST" and path == "/api/secrets":
            secret_id = self._next("secret")
            self.secrets[secret_id] = {**body, "id": secret_id}
            return httpx.Response(201, json=self.secrets[secret_id])

        if method == "GET" and path == "/api/gateway/ca":
            return httpx.Response(
                200,
                content=(
                    b"-----BEGIN CERTIFICATE-----\n"
                    b"FAKE-CA-FOR-TESTS\n"
                    b"-----END CERTIFICATE-----\n"
                ),
                headers={"Content-Type": "application/x-pem-file"},
            )

        if method == "DELETE":
            return httpx.Response(204)

        return httpx.Response(404, json={"error": "unknown route"})


def _make_agent(
    *accounts: AccountConfig,
    agent_id: str = "agent-uno",
) -> Agent:
    agent = Agent(name=agent_id, id=agent_id)
    agent.accounts = AgentAccountsConfig.model_construct(
        accounts=list(accounts),
    )
    return agent


def _service_lookup(*services: Service):
    table = {svc.name: svc for svc in services}
    return table.__getitem__


class TestRegisterAgentWithBroker:
    def test_registers_secrets_for_brokerable_accounts(self, monkeypatch):
        monkeypatch.setenv("MY_GH_TOKEN", "ghp_real_secret")
        monkeypatch.setenv("MY_GL_TOKEN", "glpat_real_secret")

        github = _FakeBrokerableService(
            service_name="github", host="api.github.com",
        )
        gitlab = _FakeBrokerableService(
            service_name="gitlab", host="gitlab.example.com", path="/api/*",
        )
        agent = _make_agent(
            _FakeAccountConfig(
                service="github",
                credentials=[Credential(kind="pat", env_var_name="MY_GH_TOKEN")],
            ),
            _FakeAccountConfig(
                service="gitlab",
                credentials=[Credential(kind="pat", env_var_name="MY_GL_TOKEN")],
            ),
        )

        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            binding = register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(github, gitlab),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )

        # Two secrets, with the literal values forwarded to the broker.
        assert len(handler.secrets) == 2
        values = {s["value"] for s in handler.secrets.values()}
        assert values == {"ghp_real_secret", "glpat_real_secret"}
        hosts = {s["hostPattern"] for s in handler.secrets.values()}
        assert hosts == {"api.github.com", "gitlab.example.com"}

        # Agent created and secrets bound.
        assert len(handler.agents) == 1
        agent_id = next(iter(handler.agents))
        assert sorted(handler.bindings[agent_id]) == sorted(handler.secrets.keys())

        # Binding has full machinery for the sandbox layer.
        assert isinstance(binding, BrokerBinding)
        assert binding.agent_id == agent_id
        assert binding.proxy_url.startswith("http://x:aoc_test_token@")
        assert len(binding.secret_ids) == 2
        env_names = sorted(name for name, _val in binding.placeholder_env)
        assert env_names == ["MY_GH_TOKEN", "MY_GL_TOKEN"]
        for _name, val in binding.placeholder_env:
            assert val.startswith("thorn-broker-placeholder-")

    def test_no_accounts_still_creates_broker_agent(self):
        agent = _make_agent()  # no accounts
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            binding = register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )
        assert binding.secret_ids == ()
        assert binding.placeholder_env == ()
        assert len(handler.agents) == 1

    def test_unknown_service_raises(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "ghp_x")
        agent = _make_agent(
            _FakeAccountConfig(
                service="missing-service",
                credentials=[Credential(kind="pat", env_var_name="MY_TOKEN")],
            ),
        )
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            with pytest.raises(BrokerError, match="not registered on the runtime"):
                register_agent_with_broker(
                    client=client, agent=agent,
                    service_lookup=_service_lookup(),
                    ca_certificate_path="/var/lib/onecli/ca.pem",
                )

    def test_non_brokerable_service_silently_skipped(self):
        # Account on a service that exists but doesn't implement
        # BrokerableService -- no plans, no secrets, but the agent
        # still gets registered.
        plain_svc = _NonBrokerableService(name="plain")
        agent = _make_agent(
            _FakeAccountConfig(service="plain", credentials=[]),
        )
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            binding = register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(plain_svc),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )
        assert binding.secret_ids == ()

    def test_missing_env_var_surfaces_clear_error(self, monkeypatch):
        monkeypatch.delenv("UNSET_TOKEN", raising=False)
        github = _FakeBrokerableService(
            service_name="github", host="api.github.com",
        )
        agent = _make_agent(
            _FakeAccountConfig(
                service="github",
                credentials=[Credential(kind="pat", env_var_name="UNSET_TOKEN")],
            ),
        )
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            with pytest.raises(BrokerError, match="UNSET_TOKEN"):
                register_agent_with_broker(
                    client=client, agent=agent,
                    service_lookup=_service_lookup(github),
                    ca_certificate_path="/var/lib/onecli/ca.pem",
                )

    def test_secret_name_includes_service_and_suffix(self, monkeypatch):
        monkeypatch.setenv("T", "ghp_x")
        github = _FakeBrokerableService(
            service_name="github", host="api.github.com",
        )
        agent = _make_agent(
            _FakeAccountConfig(
                service="github",
                credentials=[Credential(kind="pat", env_var_name="T")],
            ),
        )
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(github),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )
        names = [s["name"] for s in handler.secrets.values()]
        assert any(
            n == "agent-uno-github-fake-pat" for n in names
        ), names

    def test_agent_state_unchanged_after_registration(self, monkeypatch):
        # The new shape stores no literal values in agent state;
        # agent.accounts.accounts[i].credentials still references
        # the same env var names afterwards (no mutation).
        monkeypatch.setenv("T", "ghp_x")
        github = _FakeBrokerableService(
            service_name="github", host="api.github.com",
        )
        agent = _make_agent(
            _FakeAccountConfig(
                service="github",
                credentials=[Credential(kind="pat", env_var_name="T")],
            ),
        )
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(github),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )
        cred = agent.accounts.accounts[0].credentials[0]
        assert cred.env_var_name == "T"
        assert cred.kind == "pat"

    def test_value_transform_rewrites_registered_secret_value(
        self, monkeypatch,
    ):
        """When a plan carries a ``value_transform``, the broker driver
        must pass the transform's output as the registered secret
        value -- not the literal env var -- so the stored payload
        matches what OneCLI will inject on the wire."""
        monkeypatch.setenv("MY_TOKEN", "raw-pat-xyz")

        class _TransformingService(_FakeBrokerableService):
            def broker_credential_plans(
                self, account: AccountConfig,
            ) -> list[BrokerCredentialPlan]:
                return [
                    BrokerCredentialPlan(
                        env_var_name="MY_TOKEN",
                        host_pattern=self._host,
                        path_pattern="/*",
                        injection=HeaderInjection(
                            header_name="Authorization",
                            value_format="Basic {value}",
                        ),
                        secret_name_suffix="git-https",
                        value_transform=lambda raw: f"encoded::{raw}",
                    ),
                ]

        svc = _TransformingService(
            service_name="github", host="github.com",
        )
        agent = _make_agent(
            _FakeAccountConfig(
                service="github",
                credentials=[Credential(kind="pat", env_var_name="MY_TOKEN")],
            ),
        )
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(svc),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )
        assert len(handler.secrets) == 1
        assert next(iter(handler.secrets.values()))["value"] == (
            "encoded::raw-pat-xyz"
        )

    def test_git_extra_header_host_populates_binding(self, monkeypatch):
        """Plans with ``git_extra_header_host`` contribute
        ``(host, "Authorization: Basic <base64(x:placeholder)>")``
        entries to :attr:`BrokerBinding.git_extra_headers`, matching
        the shared placeholder the container env receives."""
        import base64

        monkeypatch.setenv("MY_TOKEN", "raw-pat-xyz")

        class _GitRoutingService(_FakeBrokerableService):
            def broker_credential_plans(
                self, account: AccountConfig,
            ) -> list[BrokerCredentialPlan]:
                return [
                    BrokerCredentialPlan(
                        env_var_name="MY_TOKEN",
                        host_pattern=self._host,
                        path_pattern="/*",
                        injection=HeaderInjection(
                            header_name="Authorization",
                            value_format="Basic {value}",
                        ),
                        secret_name_suffix="git-https",
                        value_transform=lambda raw: "pre-encoded",
                        git_extra_header_host=self._host,
                    ),
                ]

        svc = _GitRoutingService(
            service_name="github", host="github.com",
        )
        agent = _make_agent(
            _FakeAccountConfig(
                service="github",
                credentials=[Credential(kind="pat", env_var_name="MY_TOKEN")],
            ),
        )
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            binding = register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(svc),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )
        # One extra-header entry against the git host.
        assert len(binding.git_extra_headers) == 1
        host, header_value = binding.git_extra_headers[0]
        assert host == "github.com"
        # Value must be ``Authorization: Basic base64("x:<placeholder>")``
        # where ``placeholder`` is the same string injected into the
        # container's env.
        placeholder = dict(binding.placeholder_env)["MY_TOKEN"]
        expected = base64.b64encode(
            f"x:{placeholder}".encode(),
        ).decode()
        assert header_value == f"Authorization: Basic {expected}"
        # Gateway layer fills in ``git_config_path``; the broker
        # driver leaves it None so the file-rendering step stays
        # owned by a single place.
        assert binding.git_config_path is None

    def test_shared_placeholder_across_plans_for_same_env_var(
        self, monkeypatch,
    ):
        """A service returning two plans referencing the same env var
        (the real-world shape: one API plan, one git HTTPS plan)
        must produce a single placeholder_env entry.  Otherwise the
        in-container GITHUB_TOKEN and the gitconfig extraHeader's
        encoded payload would disagree and the broker rewriting the
        header value wouldn't line up with tools that also read the
        raw env var."""
        monkeypatch.setenv("MY_TOKEN", "raw-pat-xyz")

        class _TwoPlanService(_FakeBrokerableService):
            def broker_credential_plans(
                self, account: AccountConfig,
            ) -> list[BrokerCredentialPlan]:
                return [
                    BrokerCredentialPlan(
                        env_var_name="MY_TOKEN",
                        host_pattern="api.github.com",
                        path_pattern="/*",
                        injection=HeaderInjection(
                            header_name="Authorization",
                            value_format="Bearer {value}",
                        ),
                        secret_name_suffix="api",
                    ),
                    BrokerCredentialPlan(
                        env_var_name="MY_TOKEN",
                        host_pattern="github.com",
                        path_pattern="/*",
                        injection=HeaderInjection(
                            header_name="Authorization",
                            value_format="Basic {value}",
                        ),
                        secret_name_suffix="git-https",
                        value_transform=lambda raw: "encoded",
                        git_extra_header_host="github.com",
                    ),
                ]

        svc = _TwoPlanService(service_name="github", host="github.com")
        agent = _make_agent(
            _FakeAccountConfig(
                service="github",
                credentials=[Credential(kind="pat", env_var_name="MY_TOKEN")],
            ),
        )
        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            binding = register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(svc),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )
        names = [n for n, _v in binding.placeholder_env]
        assert names == ["MY_TOKEN"], (
            f"expected a single placeholder_env entry for a shared "
            f"env var across plans, got {binding.placeholder_env!r}"
        )
        # But both secrets ARE registered (one per plan).
        assert len(binding.secret_ids) == 2
        assert len(handler.secrets) == 2


# ---------------------------------------------------------------------------
# Regression: untyped account that maps to a brokerable service still works
# (covers the dogfood-transcript bug where an inferred-from-URL forge
# was unknown to the broker because it was looking at GatewayConfig
# rather than the runtime).
# ---------------------------------------------------------------------------


class TestRegressionUntypedAccount:
    def test_untyped_account_dispatches_via_runtime_lookup(self, monkeypatch):
        # Simulate the gateway-startup state where validate_agent_accounts
        # hasn't run yet (or where the test wants to verify behaviour
        # against an untyped instance directly).  The broker code reads
        # ``account.service`` to look up the service, then asks the
        # service for plans -- it doesn't look at GatewayConfig.
        monkeypatch.setenv("T", "ghp_x")
        github = _FakeBrokerableService(
            service_name="github", host="api.github.com",
        )
        agent = Agent(name="bot", id="bot")
        agent.accounts = AgentAccountsConfig.model_construct(accounts=[
            UntypedAccountConfig(
                service="github",
                credentials=[Credential(kind="pat", env_var_name="T")],
            ),
        ])

        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            binding = register_agent_with_broker(
                client=client, agent=agent,
                service_lookup=_service_lookup(github),
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )
        assert len(binding.secret_ids) == 1
