"""Brain-side client for the OneCLI credential broker (Phase D).

This module is the gateway's interface to OneCLI's admin / management
HTTP API.  It is consumed by the agent-load path: for each agent that
declares forge credentials, the gateway uses :class:`BrokerClient` to
(a) ensure an OneCLI agent identity exists, (b) register one
"secret" per credential with the appropriate host+path policy and
injection config, (c) bind those secrets to the OneCLI agent, and
(d) compose the per-agent ``HTTPS_PROXY`` URL the sandbox container
will see.

R1/R2 research notes (verified against OneCLI ``main`` at
``apps/gateway/src/inject.rs::extract_agent_token`` and
``apps/web/src/app/api/{agents,secrets}/...``):

- **Proxy auth (R1).**  OneCLI's gateway accepts
  ``Proxy-Authorization: Basic base64(<username>:<token>)``, with the
  token preferred in the password field and a dummy username (the
  GitHub/GitLab/Bitbucket convention; legacy "token in username,
  empty password" is also accepted).  Standard HTTP clients
  auto-emit this header from URL credentials in ``HTTPS_PROXY``, so
  composing ``http://x:<aoc_token>@<host>:<port>/`` is sufficient and
  there is no need for per-tool bearer-header wiring.

- **Admin API (R2).**  ``Authorization: Bearer oc_<hex>`` over the
  Next.js routes under ``/api/...``.  The endpoints we drive:

  * ``POST /api/agents``  -- create agent (returns ``{id, name, ...}``,
    *not* the access token).
  * ``POST /api/agents/{id}/regenerate-token``  -- mint / rotate the
    per-agent proxy access token (``aoc_...``); we always call this
    after creation so each agent-load gets a fresh token.
  * ``POST /api/secrets``  -- register a credential with
    ``hostPattern``/``pathPattern``/``injectionConfig``.  The
    ``injectionConfig`` decides whether OneCLI inserts a header
    (``Authorization: Bearer ...``, ``Authorization: Basic ...``) or
    a query parameter into matching outbound requests.
  * ``PUT /api/agents/{id}/secrets``  -- replace the agent's bound
    secret set with a list of secret IDs.
  * ``GET /api/gateway/ca``  -- fetch the broker's MITM CA cert
    (PEM, no auth required).

  Matching is by ``hostPattern + pathPattern``, *not* by the
  placeholder string the agent sends.  The agent-side "placeholder"
  exists only so in-container tools attempt the call (refusing on
  empty token); OneCLI never inspects it.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field

from thorn.core._credentials import (
    ServiceCredential,
    assert_no_literal_credentials,
)
from thorn.gateway._config import BrokerConfig

if TYPE_CHECKING:
    from thorn.core._account import ForgeAccountConfig
    from thorn.core._agent import Agent
    from thorn.gateway._config import GatewayConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injection config and result types
# ---------------------------------------------------------------------------


class HeaderInjection(BaseModel):
    """Inject the secret value into a request header.

    ``value_format`` is OneCLI's templating syntax: the literal token
    ``{value}`` is replaced with the decrypted secret at proxy time.
    Examples: ``"Bearer {value}"`` (PAT-style), ``"token {value}"``
    (some legacy GitHub clients), ``"Basic {value}"`` (when the
    credential is already base64-encoded user:pass).
    """

    header_name: str = Field(min_length=1)
    value_format: str = Field(default="{value}", min_length=1)


class ParamInjection(BaseModel):
    """Inject the secret value into a URL query parameter."""

    param_name: str = Field(min_length=1)
    param_format: str = Field(default="{value}", min_length=1)


InjectionConfig = HeaderInjection | ParamInjection


class AgentRegistration(BaseModel):
    """Result of registering a Thorn agent with the broker."""

    agent_id: str
    """OneCLI agent UUID (opaque to us)."""

    access_token: ServiceCredential
    """Per-agent proxy access token (``aoc_...``).

    Carries ``state="literal"`` because it is a real, freshly-minted
    secret -- it just happens to authenticate to *our* broker rather
    than to any upstream service.  The audit invariant deliberately
    tolerates this kind of broker-managed credential when the
    enclosing object is the broker binding rather than the agent's
    forge accounts.
    """


class SecretRegistration(BaseModel):
    """Result of registering one credential with the broker."""

    secret_id: str
    """OneCLI secret UUID (opaque to us)."""


# ---------------------------------------------------------------------------
# BrokerClient
# ---------------------------------------------------------------------------


class BrokerError(RuntimeError):
    """Raised when an OneCLI admin API call fails.

    Phase D's agent-load contract is "broker failures fail agent-load":
    the gateway propagates this to the operator rather than silently
    falling back to env injection, because doing so would break the
    audit invariant in a way that's hard to detect later.
    """


class BrokerClient:
    """Synchronous OneCLI admin / management API client.

    Constructed once per gateway process from a :class:`BrokerConfig`,
    then driven by the agent-load path.  Sync (rather than async)
    because agent-load itself is sync today; the broker calls are a
    handful of REST round-trips per agent at startup, not on the hot
    path.

    Use as a context manager to ensure the underlying HTTP connection
    pool is closed::

        with BrokerClient(config) as broker:
            registration = broker.register_agent(...)
            ...

    The client does not retain any per-agent state -- it is purely a
    thin wrapper over OneCLI's HTTP endpoints.  The gateway is
    responsible for tracking ``agent_id``/``secret_id`` returned from
    registration and feeding them back to ``delete_agent`` /
    ``delete_secret`` at teardown.
    """

    def __init__(
        self,
        config: BrokerConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Build the client.

        *transport* is an injection seam for tests (use
        :class:`httpx.MockTransport`).  Production callers leave it
        unset and httpx uses its default ``HTTPTransport``.
        """
        self._config = config
        self._http = httpx.Client(
            base_url=config.admin_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.admin_api_key}",
                "Accept": "application/json",
            },
            timeout=30.0,
            transport=transport,
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "BrokerClient":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

    # ── Agents ───────────────────────────────────────────────────────

    def register_agent(self, *, identifier: str, name: str) -> AgentRegistration:
        """Create the agent and mint a fresh access token in one call.

        OneCLI's ``POST /api/agents`` does *not* return the new
        agent's access token in its response (only ``id``, ``name``,
        ``identifier``, ``createdAt``).  We always follow the create
        with ``POST /api/agents/{id}/regenerate-token`` so that every
        agent-load receives a fresh token (and the previous one, if
        any, is invalidated).
        """
        agent_id = self._create_agent(identifier=identifier, name=name)
        token = self._regenerate_agent_token(agent_id)
        return AgentRegistration(agent_id=agent_id, access_token=token)

    def delete_agent(self, agent_id: str) -> None:
        """Best-effort agent teardown.

        Idempotent: a 404 is treated as success because it means the
        agent is already gone.  Any other non-2xx raises
        :class:`BrokerError`.
        """
        response = self._http.delete(f"/api/agents/{agent_id}")
        if response.status_code == 404:
            return
        self._raise_for_status(response, "delete_agent")

    def _create_agent(self, *, identifier: str, name: str) -> str:
        response = self._http.post(
            "/api/agents",
            json={"identifier": identifier, "name": name},
        )
        self._raise_for_status(response, "create_agent")
        body = response.json()
        try:
            return str(body["id"])
        except (KeyError, TypeError) as e:
            raise BrokerError(
                f"create_agent: unexpected response shape: {body!r}"
            ) from e

    def _regenerate_agent_token(self, agent_id: str) -> ServiceCredential:
        response = self._http.post(
            f"/api/agents/{agent_id}/regenerate-token",
        )
        self._raise_for_status(response, "regenerate_agent_token")
        body = response.json()
        try:
            token = str(body["accessToken"])
        except (KeyError, TypeError) as e:
            raise BrokerError(
                f"regenerate_agent_token: unexpected response shape: {body!r}"
            ) from e
        return ServiceCredential(token, state="literal")

    # ── Secrets ──────────────────────────────────────────────────────

    def register_secret(
        self,
        *,
        name: str,
        value: ServiceCredential | str,
        host_pattern: str,
        path_pattern: str | None = None,
        injection: InjectionConfig,
    ) -> SecretRegistration:
        """Register a credential with OneCLI for proxy injection.

        *value* is the actual secret -- a ``ServiceCredential`` or
        plain ``str``.  The wire body uses the underlying string; the
        ``ServiceCredential`` wrapper exists for type-tracking inside
        Thorn, not on the wire.

        *host_pattern* must be a hostname (no scheme, no path); see
        OneCLI's ``hostPatternSchema`` validation.  *path_pattern* is
        a glob-style URL path filter (e.g. ``"/*"`` to match all
        paths under the host).

        *injection* declares how the proxy splices the secret into
        outbound requests that match (host, path).
        """
        body: dict[str, Any] = {
            "name": name,
            "type": "generic",
            "value": str(value),
            "hostPattern": host_pattern,
            "injectionConfig": _injection_to_wire(injection),
        }
        if path_pattern is not None:
            body["pathPattern"] = path_pattern

        response = self._http.post("/api/secrets", json=body)
        self._raise_for_status(response, "register_secret")
        result = response.json()
        try:
            return SecretRegistration(secret_id=str(result["id"]))
        except (KeyError, TypeError) as e:
            raise BrokerError(
                f"register_secret: unexpected response shape: {result!r}"
            ) from e

    def delete_secret(self, secret_id: str) -> None:
        """Best-effort secret teardown; 404 treated as success."""
        response = self._http.delete(f"/api/secrets/{secret_id}")
        if response.status_code == 404:
            return
        self._raise_for_status(response, "delete_secret")

    # ── Bindings ─────────────────────────────────────────────────────

    def bind_secrets_to_agent(
        self,
        agent_id: str,
        secret_ids: list[str],
    ) -> None:
        """Replace the agent's bound secret set with *secret_ids*.

        OneCLI's endpoint is replace-semantics, not append.  Pass the
        complete desired set, including any existing bindings the
        caller wishes to retain.
        """
        response = self._http.put(
            f"/api/agents/{agent_id}/secrets",
            json={"secretIds": list(secret_ids)},
        )
        self._raise_for_status(response, "bind_secrets_to_agent")

    # ── CA certificate ───────────────────────────────────────────────

    def fetch_ca_certificate(self) -> bytes:
        """Download OneCLI's gateway MITM CA certificate (PEM bytes).

        The endpoint is unauthenticated server-side, but we still
        send the admin auth header for consistency with our other
        calls.  Returns the raw PEM bytes; callers persist these to
        :attr:`BrokerConfig.ca_certificate_path` for the per-container
        bind mount.
        """
        response = self._http.get("/api/gateway/ca")
        self._raise_for_status(response, "fetch_ca_certificate")
        return response.content

    # ── Proxy URL composition (R1) ───────────────────────────────────

    def proxy_url_for_agent(
        self,
        access_token: ServiceCredential | str,
    ) -> str:
        """Compose the per-agent ``HTTPS_PROXY`` URL.

        The username segment is a fixed dummy (``"x"``) and the token
        goes in the password segment; this is the GitHub/GitLab/
        Bitbucket convention OneCLI's gateway accepts.  Tools that
        respect ``HTTPS_PROXY`` (curl, git, requests, Node fetch, gh,
        glab, etc.) auto-emit ``Proxy-Authorization: Basic`` derived
        from these URL credentials, so no per-tool bearer wiring is
        needed.
        """
        return _compose_proxy_url(self._config.proxy_url, str(access_token))

    # ── Error helpers ────────────────────────────────────────────────

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        if response.is_success:
            return
        # We log only the status + a short snippet of the body --
        # OneCLI's error responses are JSON envelopes that may include
        # request echoes; including the full body unbounded here would
        # risk leaking credential material in logs in pathological
        # cases.
        snippet = response.text[:200]
        raise BrokerError(
            f"OneCLI {operation} failed: HTTP {response.status_code}: {snippet}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _injection_to_wire(injection: InjectionConfig) -> dict[str, str]:
    """Translate our snake_case injection model to OneCLI's wire keys."""
    if isinstance(injection, HeaderInjection):
        return {
            "headerName": injection.header_name,
            "valueFormat": injection.value_format,
        }
    if isinstance(injection, ParamInjection):
        return {
            "paramName": injection.param_name,
            "paramFormat": injection.param_format,
        }
    raise TypeError(f"unknown injection type: {type(injection).__name__}")


# Use a constant rather than a magic string so the username segment
# is visible at the one place its meaning matters (the comment on
# proxy_url_for_agent explains why "x").
_PROXY_USERNAME = "x"


def _compose_proxy_url(proxy_url: str, access_token: str) -> str:
    """Insert ``x:<token>`` userinfo into *proxy_url*.

    Splitting via ``urlsplit``/``urlunsplit`` rather than string
    concatenation handles the corner cases: existing userinfo on the
    proxy URL is replaced; the token is percent-encoded so that a
    token containing characters like ``/`` or ``@`` doesn't corrupt
    the netloc.
    """
    parts = urlsplit(proxy_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"BrokerConfig.proxy_url is not a full URL: {proxy_url!r}")

    # Strip any prior userinfo (we always inject our own).
    host_and_port = parts.netloc.rsplit("@", 1)[-1]

    encoded_token = quote(access_token, safe="")
    userinfo = f"{_PROXY_USERNAME}:{encoded_token}"
    new_netloc = f"{userinfo}@{host_and_port}"

    return urlunsplit(
        (parts.scheme, new_netloc, parts.path, parts.query, parts.fragment),
    )


# ---------------------------------------------------------------------------
# Per-agent broker binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrokerBinding:
    """Per-agent state captured at broker-registration time.

    Produced by :func:`register_agent_with_broker`, owned by the
    gateway, and consumed by the per-agent sandbox executor (Phase D
    work item 6) when it builds the agent's container environment.

    Concretely, the sandbox executor uses *this* object to:

    - set ``HTTPS_PROXY`` / ``HTTP_PROXY`` (from :attr:`proxy_url`)
    - bind-mount the broker CA (:attr:`ca_certificate_path`) read-only
    - inject placeholder credential env values
      (:attr:`placeholder_env`) so in-container tools get a non-empty
      token to attempt their calls with

    The binding also retains :attr:`agent_id` and :attr:`secret_ids`
    so the gateway can issue a ``DELETE /api/agents/{id}`` (and
    matching secret deletions) at shutdown -- the per-load
    create+delete lifecycle decided in the Phase D plan.

    Frozen so the binding is safe to share across the agent's
    sandbox lifetime; the underlying broker secrets do not change
    during a single load.
    """

    agent_id: str
    """OneCLI agent UUID (opaque to us)."""

    secret_ids: tuple[str, ...]
    """OneCLI secret UUIDs registered for this agent."""

    access_token: ServiceCredential
    """Per-agent proxy access token (``aoc_...``).

    Carries ``state="literal"`` because it is a real credential that
    the sandbox container's HTTPS_PROXY URL embeds.  The audit
    invariant deliberately tolerates broker-managed credentials -- it
    only enforces placeholder state on credentials reachable from the
    *agent state* surface (forge accounts), not on this binding,
    which is held by the gateway and never persisted to disk.
    """

    proxy_url: str
    """Composed ``HTTPS_PROXY``-style URL with the access token
    embedded, ready to set on the sandbox container."""

    ca_certificate_path: str
    """Host filesystem path to the broker's MITM CA cert for the
    sandbox to bind-mount read-only."""

    placeholder_env: tuple[tuple[str, str], ...]
    """``(NAME, value)`` env entries the sandbox should inject so
    that in-container tools (gh, glab, git, etc.) see a non-empty
    token value and attempt their calls.  OneCLI does not inspect
    these strings -- substitution is by host+path -- but the
    in-container tool's "have I been given a token?" check needs to
    see something."""


# ---------------------------------------------------------------------------
# Registration orchestration
# ---------------------------------------------------------------------------


_PLACEHOLDER_ENV_PREFIX = "thorn-broker-placeholder-"
"""Prefix for the random placeholder strings injected into the
container's env.  The string is purely a satisfaction value for
in-container tools; OneCLI never reads it.  We pick a recognizable
prefix so an operator inspecting a running container's env can
immediately see "this isn't a real token; the proxy is doing the
substitution".
"""


def _make_placeholder_value() -> str:
    """Return a non-empty placeholder credential value.

    The value is opaque random bytes -- the tool consuming it must
    not pattern-match on the prefix; we just want something that
    isn't empty and isn't accidentally a real credential.
    """
    return f"{_PLACEHOLDER_ENV_PREFIX}{secrets.token_urlsafe(16)}"


@dataclass(frozen=True)
class _CredentialPlan:
    """Internal: how to register one ForgeAccountConfig with the broker."""

    secret_name: str
    host_pattern: str
    path_pattern: str | None
    injection: InjectionConfig
    env_var_name: str | None
    """The env-var name in-container tools expect (e.g. ``"GITHUB_TOKEN"``).
    ``None`` when there is no canonical env var (so we just register
    the secret without adding a placeholder)."""


def _plan_for_account(
    account: "ForgeAccountConfig",
    forge_url: str,
) -> _CredentialPlan:
    """Map a forge account to a broker-registration plan.

    *forge_url* is the human-facing forge URL declared in
    ``gateway.json`` (e.g. ``"https://github.com"``).  We compute the
    hostPattern from this URL plus the credential variant.

    Phase D first-cut scope: GitHub PATs (``GitHubPatAuth``) and
    GitLab PATs (``GitLabCredentials``).  GitHub App auth
    (``GitHubAppAuth``) is rejected because OneCLI's substitution
    model handles static-token injection only -- App auth needs JWT
    signing + installation-token exchange, which is a separate
    integration concern out of scope for the first phase D pass.
    """
    from thorn.core._account import GitLabCredentials
    from thorn.tools._github_connection import GitHubAppAuth, GitHubPatAuth

    creds = account.credentials
    parsed = urlsplit(forge_url)
    host = parsed.hostname or ""

    if isinstance(creds, GitHubPatAuth):
        # GitHub.com vs Enterprise: the REST API lives at
        # ``api.github.com`` for the public host but on the same
        # hostname (under ``/api/v3/``) for GHE.  We register the API
        # host so OneCLI matches just the API path; git HTTPS auth
        # (which goes to the bare host) is handled by an additional
        # per-host registration in a future pass.
        api_host = "api.github.com" if host == "github.com" else host
        return _CredentialPlan(
            secret_name=f"{account.service}-github-pat",
            host_pattern=api_host,
            path_pattern="/*",
            injection=HeaderInjection(
                header_name="Authorization",
                value_format="Bearer {value}",
            ),
            env_var_name="GITHUB_TOKEN",
        )

    if isinstance(creds, GitLabCredentials):
        # GitLab.com and self-hosted both serve the API on the same
        # host the operator points the forge at.
        return _CredentialPlan(
            secret_name=f"{account.service}-gitlab-pat",
            host_pattern=host,
            path_pattern="/api/*",
            injection=HeaderInjection(
                header_name="Authorization",
                value_format="Bearer {value}",
            ),
            env_var_name="GITLAB_TOKEN",
        )

    if isinstance(creds, GitHubAppAuth):
        raise BrokerError(
            f"GitHub App authentication for forge {account.service!r} "
            "cannot be registered with the broker in Phase D: OneCLI's "
            "substitution model handles static tokens only, not the "
            "JWT-signing + installation-token-exchange flow App auth "
            "requires.  Either disable the broker for this agent "
            "(remove gateway.json's broker block) or migrate the "
            "account to PAT auth."
        )

    raise BrokerError(
        f"Cannot register credentials of type {type(creds).__name__} "
        f"for forge {account.service!r} with the broker"
    )


def _forge_url_for_service(
    config: "GatewayConfig",
    service_name: str,
) -> str:
    """Look up the human-facing forge URL for an agent account's service.

    The agent's ``ForgeAccountConfig.service`` field references an
    entry in ``gateway.json``'s ``forges[]`` by ``name``; that entry
    carries the URL.
    """
    for forge in config.forges:
        if forge.name == service_name:
            if not forge.url:
                raise BrokerError(
                    f"forge {service_name!r} has no URL configured in "
                    "gateway.json; cannot derive broker hostPattern"
                )
            return forge.url
    raise BrokerError(
        f"forge {service_name!r} (referenced by an agent account) is "
        f"not declared in gateway.json's forges[] array"
    )


def register_agent_with_broker(
    *,
    client: BrokerClient,
    agent: "Agent",
    config: "GatewayConfig",
    ca_certificate_path: str,
) -> BrokerBinding:
    """Register *agent*'s forge credentials with the broker.

    Performs the full Phase D agent-load registration sequence:

    1. For each forge account on *agent*: register its credential as
       an OneCLI secret (with the appropriate host+path policy and
       injection config), keeping track of the resulting secret ID.
    2. Create a fresh OneCLI agent and mint its proxy access token.
    3. Bind the registered secrets to the freshly-created agent.
    4. Replace each agent account's literal credential with a
       placeholder ``ServiceCredential`` (state ``"placeholder"``)
       so that subsequent in-process reads see only the placeholder.
       This is the swap that makes the audit invariant true.
    5. Run :func:`assert_no_literal_credentials` over the agent to
       confirm the swap fully scrubbed the agent's reachable state.

    On any broker error this function raises :class:`BrokerError`
    without attempting partial cleanup; the caller (gateway startup)
    is expected to fail-fast and surface the original error to the
    operator.  Cleanup of partial registrations falls to the
    per-load ``DELETE`` issued at shutdown.

    *ca_certificate_path* is the host filesystem path the gateway
    has already populated with the broker's MITM CA cert (one fetch
    per startup, see :meth:`Gateway._register_broker_bindings`).
    Threaded through here so the resulting :class:`BrokerBinding`
    carries the resolved path alongside the other per-agent
    sandbox-launch wiring -- the runtime then bind-mounts the same
    file into every per-agent container.  Passing the path
    explicitly (rather than reading it back off ``client._config``)
    keeps the resolution policy a single concern of the gateway,
    and makes the function trivially testable without a configured
    CA file on disk.

    Returns a :class:`BrokerBinding` containing the proxy URL, CA
    path, and placeholder env entries for the per-agent sandbox to
    consume.
    """
    if agent.id is None:
        raise BrokerError(
            "Cannot register an agent without an id with the broker"
        )

    accounts_state = getattr(agent, "accounts", None)
    forge_accounts: list[ForgeAccountConfig]
    if accounts_state is None:
        forge_accounts = []
    else:
        forge_accounts = list(accounts_state.forge_accounts())

    plans: list[tuple[ForgeAccountConfig, _CredentialPlan]] = [
        (account, _plan_for_account(
            account, _forge_url_for_service(config, account.service),
        ))
        for account in forge_accounts
    ]

    # Phase 1: register each credential as a secret.
    secret_ids: list[str] = []
    for account, plan in plans:
        # Use the actual ServiceCredential value (the literal) here;
        # this is the only point in the registration flow where the
        # literal leaves the agent's process boundary, and only over
        # the broker's TLS-bridged admin connection.
        secret_value = _credential_secret_value(account.credentials)
        registration = client.register_secret(
            name=f"{agent.id}-{plan.secret_name}",
            value=secret_value,
            host_pattern=plan.host_pattern,
            path_pattern=plan.path_pattern,
            injection=plan.injection,
        )
        secret_ids.append(registration.secret_id)

    # Phase 2: create the OneCLI agent and bind the secrets.
    agent_registration = client.register_agent(
        identifier=_broker_identifier_for_agent(agent),
        name=f"thorn:{agent.id}",
    )
    if secret_ids:
        client.bind_secrets_to_agent(
            agent_registration.agent_id, secret_ids,
        )

    # Phase 3: rewrite the agent's in-memory credentials to placeholders.
    placeholder_env: list[tuple[str, str]] = []
    for account, plan in plans:
        placeholder_value = _make_placeholder_value()
        _replace_credential_with_placeholder(account, placeholder_value)
        if plan.env_var_name is not None:
            placeholder_env.append((plan.env_var_name, placeholder_value))

    # Phase 4: audit -- the agent must hold no non-empty literal
    # credentials at this point.  Hard-fail if it does so we surface
    # programming errors loudly rather than letting a literal token
    # ride along into the container env.
    assert_no_literal_credentials(agent)

    proxy_url = client.proxy_url_for_agent(agent_registration.access_token)

    return BrokerBinding(
        agent_id=agent_registration.agent_id,
        secret_ids=tuple(secret_ids),
        access_token=agent_registration.access_token,
        proxy_url=proxy_url,
        ca_certificate_path=ca_certificate_path,
        placeholder_env=tuple(placeholder_env),
    )


def _broker_identifier_for_agent(agent: "Agent") -> str:
    """Compute a stable OneCLI agent identifier for *agent*.

    OneCLI's identifier regex is ``^[a-z][a-z0-9-]{0,49}$``.  We
    derive from the Thorn agent ID, lowercasing and replacing
    forbidden characters with hyphens, and prefixing ``"a"`` if the
    first character isn't a lowercase letter.  Truncated to 50 chars.
    """
    raw = str(agent.id).lower()
    cleaned: list[str] = []
    for ch in raw:
        if ch.isascii() and (ch.isalnum() or ch == "-"):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    sanitized = "".join(cleaned)
    if not sanitized or not (sanitized[0].isascii() and sanitized[0].isalpha()):
        sanitized = "a" + sanitized
    return sanitized[:50]


def _credential_secret_value(creds: Any) -> str:
    """Extract the underlying credential string from a forge-credential.

    Returns the string the broker should store as the secret's
    ``value`` (i.e. the real token).  Type dispatch is intentional so
    that adding a future credential type is a localized change here.
    """
    from thorn.core._account import GitLabCredentials
    from thorn.tools._github_connection import GitHubPatAuth

    if isinstance(creds, GitHubPatAuth):
        return str(creds.token)
    if isinstance(creds, GitLabCredentials):
        return str(creds.token)
    raise BrokerError(
        f"Cannot extract secret value from credentials of type "
        f"{type(creds).__name__}"
    )


def _replace_credential_with_placeholder(
    account: "ForgeAccountConfig",
    placeholder: str,
) -> None:
    """Mutate *account* in place: swap its literal credential for a
    placeholder-state ``ServiceCredential``."""
    from thorn.core._account import GitLabCredentials
    from thorn.tools._github_connection import GitHubPatAuth

    creds = account.credentials
    placeholder_cred = ServiceCredential(placeholder, state="placeholder")
    if isinstance(creds, GitHubPatAuth):
        creds.token = placeholder_cred
        return
    if isinstance(creds, GitLabCredentials):
        creds.token = placeholder_cred
        return
    raise BrokerError(
        f"Cannot rewrite placeholder for credentials of type "
        f"{type(creds).__name__}"
    )


__all__ = [
    "AgentRegistration",
    "BrokerBinding",
    "BrokerClient",
    "BrokerError",
    "HeaderInjection",
    "InjectionConfig",
    "ParamInjection",
    "SecretRegistration",
    "register_agent_with_broker",
]
