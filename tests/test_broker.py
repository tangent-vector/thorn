"""Tests for :mod:`thorn.gateway._broker` (Phase D broker client).

These tests intercept HTTP traffic with :class:`httpx.MockTransport`
so the wire shapes of every admin-API call are pinned.  R1 (proxy
URL composition) and R2 (admin endpoints, auth header, request /
response shapes) are baked into the assertions here so a future
OneCLI upstream change is a noisy test failure rather than a silent
agent-load bug.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from thorn.core._credentials import ServiceCredential
from thorn.core._account import (
    AgentAccountsConfig,
    ForgeAccountConfig,
    GitLabCredentials,
)
from thorn.core._agent import Agent
from thorn.tools._github_connection import GitHubAppAuth, GitHubPatAuth
from thorn.gateway._broker import (
    AgentRegistration,
    BrokerBinding,
    BrokerClient,
    BrokerError,
    HeaderInjection,
    ParamInjection,
    SecretRegistration,
    _broker_identifier_for_agent,
    _compose_proxy_url,
    _plan_for_account,
    register_agent_with_broker,
)
from thorn.gateway._config import BrokerConfig, ForgeSpec, GatewayConfig


def _broker_config(**overrides: object) -> BrokerConfig:
    base: dict[str, object] = {
        "admin_url": "http://onecli-web:10254",
        "admin_api_key": "oc_admin_dummy",
        "proxy_url": "http://onecli-gateway:10255",
    }
    # ca_certificate_path is gateway-resolved at startup (defaults
    # to a path under the agency home).  Client-level tests
    # exercise the BrokerClient surface directly and do not need it
    # set on the config.
    base.update(overrides)
    return BrokerConfig.model_validate(base)


def _client_with_router(
    config: BrokerConfig,
    handler: httpx._types.RequestHandler,
) -> BrokerClient:
    return BrokerClient(config, transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Auth header (R2)
# ---------------------------------------------------------------------------


class TestAuthHeader:
    def test_admin_calls_send_bearer_oc_token(self):
        captured: dict[str, str | None] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers.get("Authorization")
            return httpx.Response(200, json=[])

        config = _broker_config(admin_api_key="oc_real_key_xyz")
        with _client_with_router(config, handler) as broker:
            # Any read-only call works; pick one with a stable shape.
            broker.fetch_ca_certificate()

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
                # Wire shape that OneCLI expects.
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
        assert registration.access_token.is_literal
        # Both calls happened, in the right order.
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
            return httpx.Response(
                201,
                json={
                    "id": "secret-1",
                    "name": "ghp",
                    "type": "generic",
                    "hostPattern": "api.github.com",
                    "pathPattern": "/*",
                    "createdAt": "2026-04-28T00:00:00Z",
                    "preview": "ghp_•••••••• (redacted)",
                },
            )

        with _client_with_router(_broker_config(), handler) as broker:
            result = broker.register_secret(
                name="ghp",
                value=ServiceCredential("ghp_real_pat"),
                host_pattern="api.github.com",
                path_pattern="/*",
                injection=HeaderInjection(
                    header_name="Authorization",
                    value_format="Bearer {value}",
                ),
            )

        assert isinstance(result, SecretRegistration)
        assert result.secret_id == "secret-1"
        # Wire shape: snake_case fields are translated to OneCLI's camelCase,
        # and the credential value is unwrapped to a plain string.
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
                name="api-key",
                value="raw-string-also-works",
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
            return httpx.Response(400, json={"error": "bad request"})

        with _client_with_router(_broker_config(), handler) as broker:
            with pytest.raises(BrokerError, match="HTTP 400"):
                broker.register_secret(
                    name="x",
                    value="v",
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
# CA certificate (R2)
# ---------------------------------------------------------------------------


class TestCaCertificate:
    def test_returns_pem_bytes(self):
        pem = (
            b"-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/gateway/ca"
            return httpx.Response(
                200,
                content=pem,
                headers={"content-type": "application/x-pem-file"},
            )

        with _client_with_router(_broker_config(), handler) as broker:
            result = broker.fetch_ca_certificate()

        assert result == pem


# ---------------------------------------------------------------------------
# Proxy URL composition (R1)
# ---------------------------------------------------------------------------


class TestProxyUrlComposition:
    def test_basic_form(self):
        url = _compose_proxy_url(
            "http://broker:10255", "aoc_simple_token",
        )
        assert url == "http://x:aoc_simple_token@broker:10255"

    def test_replaces_existing_userinfo(self):
        # If the operator misconfigures the URL with stale userinfo,
        # we always replace rather than append (otherwise we end up
        # with two ``user:pass`` segments and httpx rejects it).
        url = _compose_proxy_url(
            "http://stale:creds@broker:10255", "aoc_x",
        )
        assert url == "http://x:aoc_x@broker:10255"

    def test_percent_encodes_token(self):
        # An ``aoc_`` token won't normally contain reserved characters,
        # but our compose path must still defend against it.
        url = _compose_proxy_url(
            "http://broker:10255", "aoc_a/b@c",
        )
        # ``/`` and ``@`` percent-encoded; the resulting netloc has a
        # single ``@`` separator.
        assert url == "http://x:aoc_a%2Fb%40c@broker:10255"

    def test_preserves_path_query_fragment(self):
        url = _compose_proxy_url(
            "http://broker:10255/prefix?x=1#frag", "aoc_t",
        )
        assert url == "http://x:aoc_t@broker:10255/prefix?x=1#frag"

    def test_rejects_non_url(self):
        with pytest.raises(ValueError, match="not a full URL"):
            _compose_proxy_url("just-a-string", "aoc_t")

    def test_proxy_url_for_agent_uses_config(self):
        broker = BrokerClient(
            _broker_config(proxy_url="http://onecli:10255"),
            transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        )
        try:
            url = broker.proxy_url_for_agent(
                ServiceCredential("aoc_xyz", state="literal"),
            )
        finally:
            broker.close()
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
        # Underscores and dots become hyphens to satisfy
        # OneCLI's identifier regex.
        assert _broker_identifier_for_agent(agent) == "my-agent-id"

    def test_prefixes_when_starts_with_digit(self):
        agent = Agent(name="x", id="123agent")
        # Identifier regex requires a leading lowercase letter.
        result = _broker_identifier_for_agent(agent)
        assert result.startswith("a")
        assert result == "a123agent"

    def test_truncates_to_50_chars(self):
        agent = Agent(name="x", id="a" * 100)
        result = _broker_identifier_for_agent(agent)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# Per-account plan dispatch
# ---------------------------------------------------------------------------


class TestPlanForAccount:
    def test_github_pat_plan(self):
        account = ForgeAccountConfig(
            service="github-com",
            credentials=GitHubPatAuth(token="ghp_real"),
        )
        plan = _plan_for_account(account, "https://github.com")
        assert plan.host_pattern == "api.github.com"
        assert plan.path_pattern == "/*"
        assert isinstance(plan.injection, HeaderInjection)
        assert plan.injection.header_name == "Authorization"
        assert plan.injection.value_format == "Bearer {value}"
        assert plan.env_var_name == "GITHUB_TOKEN"

    def test_github_enterprise_pat_uses_same_host(self):
        # GitHub Enterprise serves its REST API under the same
        # hostname (under /api/v3); we keep the GHE host in the
        # pattern rather than aliasing to api.github.com.
        account = ForgeAccountConfig(
            service="ghe",
            credentials=GitHubPatAuth(token="ghe_real"),
        )
        plan = _plan_for_account(account, "https://github.example.com")
        assert plan.host_pattern == "github.example.com"

    def test_gitlab_pat_plan(self):
        account = ForgeAccountConfig(
            service="gitlab",
            credentials=GitLabCredentials(token="glpat_real"),
        )
        plan = _plan_for_account(account, "https://gitlab.com")
        assert plan.host_pattern == "gitlab.com"
        assert plan.path_pattern == "/api/*"
        assert plan.env_var_name == "GITLAB_TOKEN"

    def test_github_app_auth_rejected(self):
        account = ForgeAccountConfig(
            service="github-com",
            credentials=GitHubAppAuth(
                app_id="123",
                installation_id=456,
                private_key_pem="-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
            ),
        )
        with pytest.raises(BrokerError, match="GitHub App authentication"):
            _plan_for_account(account, "https://github.com")


# ---------------------------------------------------------------------------
# End-to-end registration orchestration
# ---------------------------------------------------------------------------


def _gateway_config_with_forges(*forges: ForgeSpec) -> GatewayConfig:
    cfg = GatewayConfig(forges=list(forges))
    # Trigger the model validators that fill in derived fields.
    return GatewayConfig.model_validate(cfg.model_dump())


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
        import json

        path = request.url.path
        method = request.method
        body = (
            json.loads(request.content.decode("utf-8"))
            if request.content else {}
        )

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
            # Phase D: gateway fetches the broker's CA from this
            # endpoint at startup and writes the bytes to disk.
            # Tests that drive ``Gateway._register_broker_bindings``
            # need the handler to answer this call (otherwise the
            # gateway hits a 404 and bails before any agent
            # registration happens).
            return httpx.Response(
                200,
                content=(
                    b"-----BEGIN CERTIFICATE-----\n"
                    b"FAKE-CA-FOR-TESTS\n"
                    b"-----END CERTIFICATE-----\n"
                ),
                headers={"Content-Type": "application/x-pem-file"},
            )

        return httpx.Response(404, json={"error": "unknown route"})


class TestRegisterAgentWithBroker:
    def _make_agent(self, *accounts: ForgeAccountConfig) -> Agent:
        agent = Agent(name="agent-1", id="agent-uno")
        agent.accounts = AgentAccountsConfig(accounts=list(accounts))
        return agent

    def test_registers_secrets_creates_agent_binds_and_swaps(self):
        gateway_cfg = _gateway_config_with_forges(
            ForgeSpec(name="github-com", type="github", url="https://github.com"),
            ForgeSpec(name="my-gitlab", type="gitlab", url="https://gitlab.example.com"),
        )
        agent = self._make_agent(
            ForgeAccountConfig(
                service="github-com",
                credentials=GitHubPatAuth(token="ghp_real_secret"),
            ),
            ForgeAccountConfig(
                service="my-gitlab",
                credentials=GitLabCredentials(token="glpat_real_secret"),
            ),
        )

        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            binding = register_agent_with_broker(
                client=client, agent=agent, config=gateway_cfg,
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )

        # Two secrets registered.
        assert len(handler.secrets) == 2
        gh_secret = next(s for s in handler.secrets.values()
                         if "github-pat" in s["name"])
        gl_secret = next(s for s in handler.secrets.values()
                         if "gitlab-pat" in s["name"])
        # The literal credential VALUE was sent to the broker (this
        # is the legitimate egress point).
        assert gh_secret["value"] == "ghp_real_secret"
        assert gl_secret["value"] == "glpat_real_secret"
        # Host patterns derived from forge URLs.
        assert gh_secret["hostPattern"] == "api.github.com"
        assert gl_secret["hostPattern"] == "gitlab.example.com"

        # Agent created and secrets bound.
        assert len(handler.agents) == 1
        agent_id = next(iter(handler.agents))
        assert sorted(handler.bindings[agent_id]) == sorted(
            handler.secrets.keys(),
        )

        # Agent's in-memory credentials are now placeholders.
        for account in agent.accounts.forge_accounts():
            cred = account.credentials
            token = (
                cred.token if hasattr(cred, "token")
                else None  # pragma: no cover -- only PAT/GitLab in this test
            )
            assert isinstance(token, ServiceCredential)
            assert token.is_placeholder
            assert "thorn-broker-placeholder-" in str(token)

        # Binding has full machinery for the sandbox layer.
        assert isinstance(binding, BrokerBinding)
        assert binding.agent_id == agent_id
        assert binding.proxy_url.startswith("http://x:aoc_test_token@")
        assert len(binding.secret_ids) == 2
        env_names = [name for name, _val in binding.placeholder_env]
        assert sorted(env_names) == ["GITHUB_TOKEN", "GITLAB_TOKEN"]
        for _name, val in binding.placeholder_env:
            assert val.startswith("thorn-broker-placeholder-")

    def test_no_accounts_still_creates_broker_agent(self):
        # An agent with no forge accounts still gets a broker
        # binding (so its sandbox can route through the proxy for
        # things like LLM API calls registered at the agency level
        # in a future phase).  Empty secret list, empty placeholder
        # env.
        gateway_cfg = _gateway_config_with_forges()
        agent = self._make_agent()  # no accounts

        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            binding = register_agent_with_broker(
                client=client, agent=agent, config=gateway_cfg,
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )

        assert binding.secret_ids == ()
        assert binding.placeholder_env == ()
        # Agent still created on the broker.
        assert len(handler.agents) == 1

    def test_unknown_forge_raises(self):
        # The agent references a forge that isn't in gateway.json.
        gateway_cfg = _gateway_config_with_forges()
        agent = self._make_agent(
            ForgeAccountConfig(
                service="missing-forge",
                credentials=GitHubPatAuth(token="ghp_x"),
            ),
        )

        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            with pytest.raises(BrokerError, match="not declared"):
                register_agent_with_broker(
                    client=client, agent=agent, config=gateway_cfg,
                    ca_certificate_path="/var/lib/onecli/ca.pem",
                )

    def test_app_auth_blocks_registration(self):
        gateway_cfg = _gateway_config_with_forges(
            ForgeSpec(name="github-com", type="github", url="https://github.com"),
        )
        agent = self._make_agent(
            ForgeAccountConfig(
                service="github-com",
                credentials=GitHubAppAuth(
                    app_id="123",
                    installation_id=456,
                    private_key_pem="-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
                ),
            ),
        )

        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            with pytest.raises(BrokerError, match="GitHub App"):
                register_agent_with_broker(
                    client=client, agent=agent, config=gateway_cfg,
                    ca_certificate_path="/var/lib/onecli/ca.pem",
                )

    def test_audit_invariant_holds_post_registration(self):
        from thorn.core._credentials import assert_no_literal_credentials

        gateway_cfg = _gateway_config_with_forges(
            ForgeSpec(name="github-com", type="github", url="https://github.com"),
        )
        agent = self._make_agent(
            ForgeAccountConfig(
                service="github-com",
                credentials=GitHubPatAuth(token="ghp_real"),
            ),
        )

        handler = _RegistrationHandler()
        with _client_with_router(_broker_config(), handler) as client:
            register_agent_with_broker(
                client=client, agent=agent, config=gateway_cfg,
                ca_certificate_path="/var/lib/onecli/ca.pem",
            )

        # If any literal credential survived, this would raise.
        assert_no_literal_credentials(agent)


# ---------------------------------------------------------------------------
# Gateway-level broker hooks
# ---------------------------------------------------------------------------


class TestGatewayBrokerHooks:
    """Tests of the Gateway's :meth:`_register_broker_bindings` and
    :meth:`_teardown_broker_bindings` hooks.

    These tests bypass the full ``_startup`` machinery (runtime,
    scheduler creation, session-store traversal) and drive the
    broker hooks directly with a stubbed scheduler holding a
    pre-built :class:`Agent`.  The end-to-end registration logic is
    covered in :class:`TestRegisterAgentWithBroker` above; what we
    need from the gateway level is just that the hook iterates over
    schedulers, calls registration in a worker thread, populates
    bindings, and tears them down on shutdown.
    """

    def _make_gateway(
        self,
        tmp_path: Any,
        *,
        broker_block: BrokerConfig | None,
        agent_accounts: AgentAccountsConfig,
        broker_client_factory: Any = None,
    ) -> Any:
        # Imports kept inside the helper so the broker-test module
        # doesn't pay the runtime / gateway import cost when
        # collecting tests that don't need them.
        from unittest.mock import MagicMock

        from thorn.core._provider import MockProvider
        from thorn.gateway._config import SandboxConfig
        from thorn.gateway._gateway import Gateway
        from thorn.runtime import AgentID, Runtime

        gateway_config = _gateway_config_with_forges(
            ForgeSpec(name="github-com", type="github", url="https://github.com"),
            ForgeSpec(name="my-gitlab", type="gitlab", url="https://gitlab.example.com"),
        )
        # Splice the broker block onto the gateway config.
        gateway_config = GatewayConfig.model_validate(
            {**gateway_config.model_dump(), "broker": (
                broker_block.model_dump() if broker_block is not None else None
            )},
        )

        # Phase D: ``_register_broker_bindings`` short-circuits when
        # the sandbox backend is not 'container' (subprocess + broker
        # would swap real creds for placeholders without anywhere to
        # inject them).  Tests in this class exercise the
        # registration path itself, so they hand the runtime a
        # container-mode :class:`SandboxConfig` to clear that gate;
        # the broker's downstream interaction with the sandbox is
        # covered by the sandbox-side tests in
        # :mod:`tests.sandbox.test_runtime_wiring`.
        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
            sandbox_config=SandboxConfig(
                backend="container", image="thorn-sandbox:test",
            ),
        )
        gateway = Gateway(
            runtime=runtime,
            sources=[],
            gateway_config=gateway_config,
            broker_client_factory=broker_client_factory,
        )

        agent = Agent(name="test-agent", id="test-agent-id")
        agent.accounts = agent_accounts
        scheduler = MagicMock()
        scheduler.agent = agent
        gateway._schedulers[AgentID("test-agent-id")] = scheduler

        return gateway, agent

    @pytest.mark.asyncio
    async def test_no_op_when_gateway_config_absent(self, tmp_path: Any) -> None:
        from thorn.core._provider import MockProvider
        from thorn.gateway._gateway import Gateway
        from thorn.runtime import Runtime

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        gateway = Gateway(runtime=runtime, sources=[])  # no gateway_config

        await gateway._register_broker_bindings()
        assert gateway._broker_bindings == {}
        assert gateway._broker_client is None

    @pytest.mark.asyncio
    async def test_no_op_when_broker_disabled(self, tmp_path: Any) -> None:
        broker = BrokerConfig.model_validate(
            {**_broker_dict(), "enabled": False},
        )
        gateway, _agent = self._make_gateway(
            tmp_path, broker_block=broker,
            agent_accounts=AgentAccountsConfig(),
        )

        await gateway._register_broker_bindings()
        assert gateway._broker_bindings == {}

    @pytest.mark.asyncio
    async def test_no_op_when_subprocess_backend_with_warning(
        self, tmp_path: Any, caplog: Any,
    ) -> None:
        """Phase D: broker is conditional on the container backend.

        When the agency's sandbox backend resolves to subprocess (or
        is omitted entirely), ``_register_broker_bindings`` skips
        registration and emits a warning so the operator notices
        the configuration mismatch.  This is the safety valve for
        the policy 'broker integration is conditional on the
        container backend' from the Phase D plan.
        """
        import logging

        from thorn.core._provider import MockProvider
        from thorn.gateway._config import GatewayConfig, SandboxConfig
        from thorn.gateway._gateway import Gateway
        from thorn.runtime import AgentID, Runtime

        gateway_config = _gateway_config_with_forges()
        gateway_config = GatewayConfig.model_validate(
            {**gateway_config.model_dump(), "broker": _broker_dict()},
        )

        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
            sandbox_config=SandboxConfig(backend="subprocess"),
        )
        gateway = Gateway(
            runtime=runtime, sources=[], gateway_config=gateway_config,
        )
        agent = Agent(name="a", id="a-id")
        agent.accounts = AgentAccountsConfig(
            accounts=[
                ForgeAccountConfig(
                    service="github-com",
                    credentials=GitHubPatAuth(token="ghp_should_not_register"),
                ),
            ],
        )
        from unittest.mock import MagicMock
        scheduler = MagicMock()
        scheduler.agent = agent
        gateway._schedulers[AgentID("a-id")] = scheduler

        caplog.set_level(logging.WARNING, logger="thorn.gateway._gateway")
        await gateway._register_broker_bindings()

        assert gateway._broker_bindings == {}
        assert gateway._broker_client is None
        # The agent's literal credential is *not* swapped, since
        # registration was skipped: subprocess-mode tools rely on
        # the in-process creds today.
        token = agent.accounts.accounts[0].credentials.token
        assert str(token) == "ghp_should_not_register"

        warnings = [
            r for r in caplog.records
            if "subprocess" in r.getMessage()
            and "Broker is enabled" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            f"expected one subprocess/broker mismatch warning, got "
            f"{[r.getMessage() for r in warnings]}"
        )

    @pytest.mark.asyncio
    async def test_registers_each_scheduled_agent(self, tmp_path: Any) -> None:
        from thorn.runtime import AgentID

        broker = BrokerConfig.model_validate(_broker_dict())
        accounts = AgentAccountsConfig(
            accounts=[
                ForgeAccountConfig(
                    service="github-com",
                    credentials=GitHubPatAuth(token="ghp_for_test"),
                ),
            ],
        )

        handler = _RegistrationHandler()
        factory = lambda cfg: BrokerClient(  # noqa: E731 -- one-shot
            cfg.broker, transport=httpx.MockTransport(handler),
        )
        gateway, agent = self._make_gateway(
            tmp_path, broker_block=broker,
            agent_accounts=accounts,
            broker_client_factory=factory,
        )

        await gateway._register_broker_bindings()

        binding = gateway.broker_binding_for(AgentID("test-agent-id"))
        assert binding is not None
        assert binding.proxy_url.startswith("http://x:aoc_test_token@")
        assert len(binding.secret_ids) == 1
        # In-memory credentials swapped to placeholders.
        cred_after = agent.accounts.forge_accounts()[0].credentials
        assert isinstance(cred_after.token, ServiceCredential)
        assert cred_after.token.is_placeholder

    @pytest.mark.asyncio
    async def test_teardown_deletes_agent_and_secrets(self, tmp_path: Any) -> None:
        broker = BrokerConfig.model_validate(_broker_dict())
        accounts = AgentAccountsConfig(
            accounts=[
                ForgeAccountConfig(
                    service="github-com",
                    credentials=GitHubPatAuth(token="ghp_for_test"),
                ),
            ],
        )

        delete_calls: list[tuple[str, str]] = []

        class _RecordingHandler(_RegistrationHandler):
            def __call__(self, request: httpx.Request) -> httpx.Response:
                if request.method == "DELETE":
                    delete_calls.append((request.method, request.url.path))
                    return httpx.Response(204)
                return super().__call__(request)

        handler = _RecordingHandler()
        factory = lambda cfg: BrokerClient(  # noqa: E731
            cfg.broker, transport=httpx.MockTransport(handler),
        )
        gateway, _agent = self._make_gateway(
            tmp_path, broker_block=broker,
            agent_accounts=accounts,
            broker_client_factory=factory,
        )
        await gateway._register_broker_bindings()
        assert gateway._broker_bindings  # sanity check

        await gateway._teardown_broker_bindings()
        # One agent delete + one secret delete.
        delete_paths = [path for _method, path in delete_calls]
        assert any(path.startswith("/api/agents/") for path in delete_paths)
        assert any(path.startswith("/api/secrets/") for path in delete_paths)
        # Bindings cleared even on success.
        assert gateway._broker_bindings == {}
        assert gateway._broker_client is None

    @pytest.mark.asyncio
    async def test_registration_failure_cleans_up_partial_state(
        self, tmp_path: Any,
    ) -> None:
        # If the broker errors mid-registration, anything we did
        # successfully register so far must be torn down before the
        # exception propagates -- otherwise an operator's next
        # gateway start picks up "duplicate identifier" errors.
        broker = BrokerConfig.model_validate(_broker_dict())

        # An agent with one account whose forge isn't in
        # gateway.json -- registration will raise on the lookup.
        bad_accounts = AgentAccountsConfig(
            accounts=[
                ForgeAccountConfig(
                    service="missing-forge",
                    credentials=GitHubPatAuth(token="ghp_x"),
                ),
            ],
        )

        handler = _RegistrationHandler()
        factory = lambda cfg: BrokerClient(  # noqa: E731
            cfg.broker, transport=httpx.MockTransport(handler),
        )
        gateway, _agent = self._make_gateway(
            tmp_path, broker_block=broker,
            agent_accounts=bad_accounts,
            broker_client_factory=factory,
        )

        with pytest.raises(BrokerError, match="not declared"):
            await gateway._register_broker_bindings()
        # After the failure, the gateway is in a clean state ready
        # for the next attempt.
        assert gateway._broker_bindings == {}
        assert gateway._broker_client is None


# ---------------------------------------------------------------------------
# CA acquisition (gateway-resolved path + fetch-at-startup)
# ---------------------------------------------------------------------------


class TestBrokerCAAcquisition:
    """Phase D follow-up: the gateway acquires the broker's MITM CA
    via ``GET /api/gateway/ca`` at startup and writes it to a path
    it controls.

    Two paths matter:

    * **Default** (``ca_certificate_path`` unset in ``gateway.json``)
      -- the gateway derives a path under the agency home so no
      operator-side volume / shared-mount wiring is needed.  This
      is the supported default for the host-gateway deployment
      mode (Brev VM, single-host operator).
    * **Operator override** -- when ``ca_certificate_path`` is set,
      the gateway honours it (e.g. for an operator who wants the
      CA at a specific spot to share with non-Thorn tooling).
    """

    def _gateway_with_broker(
        self,
        tmp_path: Any,
        *,
        ca_certificate_path: str | None,
    ):
        from unittest.mock import MagicMock

        from thorn.core._provider import MockProvider
        from thorn.gateway._config import GatewayConfig, SandboxConfig
        from thorn.gateway._gateway import Gateway
        from thorn.runtime import AgentID, Runtime

        gateway_config = _gateway_config_with_forges()
        broker_overrides: dict[str, object] = {}
        if ca_certificate_path is not None:
            broker_overrides["ca_certificate_path"] = ca_certificate_path
        gateway_config = GatewayConfig.model_validate(
            {
                **gateway_config.model_dump(),
                "broker": _broker_dict(**broker_overrides),
            },
        )
        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
            sandbox_config=SandboxConfig(
                backend="container", image="thorn-sandbox:test",
            ),
        )

        handler = _RegistrationHandler()
        factory = lambda cfg: BrokerClient(  # noqa: E731
            cfg.broker, transport=httpx.MockTransport(handler),
        )
        gateway = Gateway(
            runtime=runtime, sources=[],
            gateway_config=gateway_config,
            broker_client_factory=factory,
        )

        # No agents needed: CA fetch happens regardless of how
        # many schedulers exist.  Add one stub so the registration
        # loop doesn't short-circuit on "no schedulers".
        agent = Agent(name="ca-test-agent", id="ca-test-agent-id")
        scheduler = MagicMock()
        scheduler.agent = agent
        gateway._schedulers[AgentID("ca-test-agent-id")] = scheduler
        return gateway, runtime

    @pytest.mark.asyncio
    async def test_default_path_is_under_agency_home(
        self, tmp_path: Any,
    ) -> None:
        gateway, runtime = self._gateway_with_broker(
            tmp_path, ca_certificate_path=None,
        )
        await gateway._register_broker_bindings()

        expected = runtime.paths.home_root / "onecli-ca.pem"
        assert expected.is_file()
        assert expected.read_bytes().startswith(
            b"-----BEGIN CERTIFICATE-----",
        )

        # Every binding should reference the same resolved path.
        bindings = list(gateway._broker_bindings.values())
        assert bindings, "expected at least one binding"
        for binding in bindings:
            assert binding.ca_certificate_path == str(expected)

    @pytest.mark.asyncio
    async def test_operator_override_wins(
        self, tmp_path: Any,
    ) -> None:
        custom = tmp_path / "operator-ca-dir" / "broker.pem"
        gateway, _runtime = self._gateway_with_broker(
            tmp_path, ca_certificate_path=str(custom),
        )
        await gateway._register_broker_bindings()

        assert custom.is_file(), (
            "operator-supplied ca_certificate_path should be honoured "
            "verbatim (parent dir created if missing)"
        )
        bindings = list(gateway._broker_bindings.values())
        assert bindings
        for binding in bindings:
            assert binding.ca_certificate_path == str(custom)

    @pytest.mark.asyncio
    async def test_ca_fetch_failure_aborts_registration(
        self, tmp_path: Any,
    ) -> None:
        """A CA-fetch failure short-circuits before any agent
        registration runs (no half-registered agents on the broker)."""
        from unittest.mock import MagicMock

        from thorn.core._provider import MockProvider
        from thorn.gateway._config import GatewayConfig, SandboxConfig
        from thorn.gateway._gateway import Gateway
        from thorn.runtime import AgentID, Runtime

        class _NoCAHandler(_RegistrationHandler):
            def __call__(self, request: httpx.Request) -> httpx.Response:
                if request.url.path == "/api/gateway/ca":
                    return httpx.Response(500, json={"error": "broken"})
                return super().__call__(request)

        gateway_config = _gateway_config_with_forges()
        gateway_config = GatewayConfig.model_validate(
            {**gateway_config.model_dump(), "broker": _broker_dict()},
        )
        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
            sandbox_config=SandboxConfig(
                backend="container", image="thorn-sandbox:test",
            ),
        )

        handler = _NoCAHandler()
        factory = lambda cfg: BrokerClient(  # noqa: E731
            cfg.broker, transport=httpx.MockTransport(handler),
        )
        gateway = Gateway(
            runtime=runtime, sources=[],
            gateway_config=gateway_config,
            broker_client_factory=factory,
        )
        agent = Agent(name="x", id="x-id")
        scheduler = MagicMock()
        scheduler.agent = agent
        gateway._schedulers[AgentID("x-id")] = scheduler

        with pytest.raises(BrokerError, match="fetch_ca_certificate"):
            await gateway._register_broker_bindings()

        # No agent registrations should have been attempted.
        assert handler.agents == {}
        assert gateway._broker_bindings == {}


# ---------------------------------------------------------------------------
# Phase D end-to-end audit flow
# ---------------------------------------------------------------------------


class TestPhaseDAuditFlow:
    """End-to-end test for the Phase D pipeline.

    Drives the full chain in one test:

    1. Gateway-level broker registration (mock OneCLI).
    2. Agent's literal credential is replaced with a placeholder
       :class:`ServiceCredential` (D8 audit invariant).
    3. :func:`assert_no_literal_credentials` passes on the
       post-registration agent state.
    4. The runtime's broker-binding lookup returns the binding for
       the registered agent.
    5. The per-agent ``ContainerHostConfig`` built by the runtime
       carries the broker proxy URL, the CA mount, and the
       placeholder env entries.

    The point of having one test that drives all five steps is that
    a refactor of any single component (the ``ServiceCredential``
    state machine, the registration loop, the runtime lookup, the
    container spec assembly) that breaks the chain shows up here as
    a failure that calls out the chain's break point, rather than
    only at the unit boundary.

    The opt-in real-runtime smoke equivalent (an actual sandbox
    container hitting an actual broker and observing a credential
    substitution) lives behind a ``requires_podman`` mark in
    ``tests/sandbox/test_smoke_real_oci.py``; that is out of scope
    for the in-tree unit suite.
    """

    @pytest.mark.asyncio
    async def test_register_then_lookup_then_build_container_spec(
        self, tmp_path: Any,
    ) -> None:
        from unittest.mock import MagicMock

        from thorn.core._credentials import assert_no_literal_credentials
        from thorn.core._provider import MockProvider
        from thorn.gateway._config import (
            GatewayConfig, SandboxConfig,
        )
        from thorn.gateway._gateway import Gateway
        from thorn.runtime import AgentID, Runtime
        from thorn.sandbox import (
            ContainerDaemonHost,
            FakeOCIRuntimeAdapter,
        )
        from thorn.sandbox._container import (
            CONTAINER_BROKER_CA_PATH,
            NO_PROXY_DEFAULT,
        )

        gateway_config = _gateway_config_with_forges(
            ForgeSpec(
                name="github-com",
                type="github",
                url="https://github.com",
            ),
            ForgeSpec(
                name="my-gitlab",
                type="gitlab",
                url="https://gitlab.example.com",
            ),
        )
        gateway_config = GatewayConfig.model_validate(
            {**gateway_config.model_dump(), "broker": _broker_dict()},
        )

        adapter = FakeOCIRuntimeAdapter(
            present_images=["thorn-sandbox:test"],
        )
        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
            sandbox_config=SandboxConfig(
                backend="container",
                image="thorn-sandbox:test",
                egress_network="thorn-broker",
            ),
            sandbox_executor_enabled=True,
            oci_runtime_adapter=adapter,
        )

        handler = _RegistrationHandler()
        factory = lambda cfg: BrokerClient(  # noqa: E731 -- one-shot
            cfg.broker, transport=httpx.MockTransport(handler),
        )
        gateway = Gateway(
            runtime=runtime, sources=[],
            gateway_config=gateway_config,
            broker_client_factory=factory,
        )

        agent = Agent(name="audit-agent", id="audit-agent-id")
        agent.accounts = AgentAccountsConfig(
            accounts=[
                ForgeAccountConfig(
                    service="github-com",
                    credentials=GitHubPatAuth(token="ghp_real_secret_xyz"),
                ),
                ForgeAccountConfig(
                    service="my-gitlab",
                    credentials=GitLabCredentials(
                        token="glpat_real_secret_abc",
                    ),
                ),
            ],
        )
        scheduler = MagicMock()
        scheduler.agent = agent
        gateway._schedulers[AgentID("audit-agent-id")] = scheduler

        await gateway._register_broker_bindings()
        # After registration: install the lookup the way ``_startup``
        # does, so the runtime can find this agent's binding.
        runtime.set_sandbox_broker_binding_lookup(
            gateway.broker_binding_for,
        )

        # 1. Binding exists with the expected wire shape.
        binding = gateway.broker_binding_for(AgentID("audit-agent-id"))
        assert binding is not None
        assert binding.proxy_url.startswith("http://x:aoc_test_token@")
        assert len(binding.secret_ids) == 2
        assert len(binding.placeholder_env) == 2
        env_names = {name for name, _value in binding.placeholder_env}
        assert env_names == {"GITHUB_TOKEN", "GITLAB_TOKEN"}

        # 2. + 3. Audit invariant: every literal credential was
        # replaced with a placeholder.
        for account in agent.accounts.forge_accounts():
            cred = account.credentials.token
            assert isinstance(cred, ServiceCredential)
            assert cred.is_placeholder, (
                f"credential for {account.service} should be a "
                f"placeholder post-registration, got {cred!r}"
            )
        assert_no_literal_credentials(agent)

        # 4. + 5. The runtime, asked to build a sandbox executor for
        # this agent, picks up the binding and assembles a
        # ``ContainerHostConfig`` carrying the broker wiring.
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, ContainerDaemonHost)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.broker_proxy_url == binding.proxy_url
        # Default CA path is under the agency home (no
        # ``ca_certificate_path`` set in the broker block); the
        # gateway pulls the CA via ``GET /api/gateway/ca`` and
        # writes it there.  Asserting the relative shape rather
        # than the absolute tmp_path keeps this readable.
        assert cfg.broker_ca_host_path is not None
        assert cfg.broker_ca_host_path.name == "onecli-ca.pem"
        assert cfg.broker_ca_host_path.is_file(), (
            "the gateway should have fetched and written the CA "
            f"at {cfg.broker_ca_host_path} during registration"
        )
        ca_bytes = cfg.broker_ca_host_path.read_bytes()
        assert ca_bytes.startswith(b"-----BEGIN CERTIFICATE-----")
        assert cfg.broker_placeholder_env == binding.placeholder_env
        assert cfg.egress_network == "thorn-broker"

        # And finally the actual ContainerSpec the adapter would
        # see has every broker-derived knob in place.
        await executor.host.start()
        try:
            spec = adapter.container_spec(cfg.container_name)
            env_dict = dict(spec.env)
            assert env_dict["HTTPS_PROXY"] == binding.proxy_url
            assert env_dict["NO_PROXY"] == NO_PROXY_DEFAULT
            assert env_dict["REQUESTS_CA_BUNDLE"] == CONTAINER_BROKER_CA_PATH
            for name, value in binding.placeholder_env:
                assert env_dict[name] == value
            ca_mounts = [
                m for m in spec.mounts
                if str(m.target) == CONTAINER_BROKER_CA_PATH
            ]
            assert len(ca_mounts) == 1
            assert ca_mounts[0].read_only is True
            assert "--network" in spec.extra_run_args
            i = spec.extra_run_args.index("--network")
            assert spec.extra_run_args[i + 1] == "thorn-broker"
        finally:
            await executor.host.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broker_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "admin_url": "http://onecli-web:10254",
        "admin_api_key": "oc_admin_dummy",
        "proxy_url": "http://onecli-gateway:10255",
    }
    # ca_certificate_path is intentionally omitted from the default
    # so gateway-level tests exercise the "gateway-resolved path
    # under agency home" path -- the production-default codepath.
    # Tests that need a specific path can override.
    base.update(overrides)
    return base


def _read_json(request: httpx.Request) -> dict[str, Any]:
    import json
    return json.loads(request.content.decode("utf-8")) if request.content else {}
