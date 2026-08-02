"""Brain-side client for the OneCLI credential broker (Phase D).

This module is the gateway's interface to OneCLI's admin / management
HTTP API.  It is consumed by the agent-load path: for each agent that
declares accounts on services that participate in upstream credential
brokering (i.e. services implementing
:class:`thorn.core._brokering.BrokerableService`), the gateway uses
:class:`BrokerClient` to (a) ensure an OneCLI agent identity exists,
(b) register one "secret" per credential the service plans, with the
appropriate host+path policy and injection config, (c) bind those
secrets to the OneCLI agent, and (d) compose the per-agent
``HTTPS_PROXY`` URL the sandbox container will see.

The broker code is deliberately ignorant of which services exist or
what credential shapes they accept; the
:meth:`BrokerableService.broker_credential_plans` protocol pushes
that knowledge into the service module.  Adding a new
broker-routed service is a purely local change in the service
module, no edits required here.

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

import base64
import logging
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel

from thorn._redaction import REDACTED_SECRET, redact_secret_snippet, redact_secrets
from thorn.core._brokering import (
    BrokerableService,
    BrokerCredentialPlan,
    HeaderInjection,
    InjectionConfig,
    ParamInjection,
)
from thorn.core._credentials import (
    Credential,
    CredentialMissingError,
    ServiceCredential,
)
from thorn.gateway._config import BrokerConfig

if TYPE_CHECKING:
    from thorn.core._agent import Agent
    from thorn.core._service import Service

log = logging.getLogger(__name__)


# Re-exported so existing callers (and tests) that import the
# injection types from here keep working without ten-line shims; the
# canonical home is :mod:`thorn.core._brokering`.
__injection_reexports__ = (HeaderInjection, ParamInjection, InjectionConfig)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class AgentRegistration(BaseModel):
    """Result of registering a Thorn agent with the broker."""

    agent_id: str
    """OneCLI agent UUID (opaque to us)."""

    access_token: ServiceCredential
    """Per-agent proxy access token (``aoc_...``).

    A real, freshly-minted secret -- it just happens to authenticate
    to *our* broker rather than to any upstream service.  The
    :class:`ServiceCredential` wrapper exists so logging surfaces
    redact this value.
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
    isolation invariant in a way that's hard to detect later.
    """


class BrokerClient:
    """Synchronous OneCLI admin / management API client.

    Constructed once per gateway process from a :class:`BrokerConfig`
    plus the literal admin API key (read from ``os.environ`` by the
    gateway), then driven by the agent-load path.  Sync (rather than
    async) because agent-load itself is sync today; the broker calls
    are a handful of REST round-trips per agent at startup, not on
    the hot path.

    Use as a context manager to ensure the underlying HTTP connection
    pool is closed::

        with BrokerClient(config, admin_api_key=key) as broker:
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
        admin_api_key: ServiceCredential,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Build the client.

        *admin_api_key* is the literal Bearer token (already read
        from ``os.environ`` by the gateway) used as the
        ``Authorization`` header value.  Wrapping in
        :class:`ServiceCredential` keeps log surfaces from leaking
        the value.

        *transport* is an injection seam for tests (use
        :class:`httpx.MockTransport`).  Production callers leave it
        unset and httpx uses its default ``HTTPTransport``.

        Refuses to build when *config* is missing the bits we need
        (``admin_url`` empty).  Reaching this guard indicates a
        wiring bug -- for example, instantiating the client outside
        the supervised startup path -- so we surface it loudly
        rather than silently sending requests to an empty base URL.
        """
        if not config.admin_url:
            raise BrokerError(
                "BrokerClient requires admin_url to be populated; got "
                f"admin_url={config.admin_url!r}.  This typically means "
                "the bundled-broker supervisor has not run yet, or "
                "the external broker config is missing required fields."
            )
        if not admin_api_key:
            raise BrokerError(
                "BrokerClient requires a non-empty admin API key."
            )
        self._config = config
        self._admin_api_key = admin_api_key
        self._http = httpx.Client(
            base_url=config.admin_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {admin_api_key}",
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
                "create_agent: unexpected response shape: "
                f"{redact_secrets(repr(body))}"
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
                "regenerate_agent_token: unexpected response shape: "
                f"{redact_secrets(repr(body))}"
            ) from e
        return ServiceCredential(token)

    # ── Secrets ──────────────────────────────────────────────────────

    def register_secret(
        self,
        *,
        name: str,
        value: str,
        host_pattern: str,
        path_pattern: str | None = None,
        injection: InjectionConfig,
    ) -> SecretRegistration:
        """Register a credential with OneCLI for proxy injection.

        *value* is the actual secret string to forward to the broker
        (passed as a plain ``str`` so callers can't accidentally
        retain a reference inside agent state).

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
            "value": value,
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
                "register_secret: unexpected response shape: "
                f"{redact_secrets(repr(result))}"
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
        snippet = redact_secret_snippet(response.text, max_chars=200)
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
    gateway, and consumed by the per-agent sandbox executor when it
    builds the agent's container environment.

    Concretely, the sandbox executor uses *this* object to:

    - set ``HTTPS_PROXY`` / ``HTTP_PROXY`` (from :attr:`proxy_url`)
    - bind-mount the broker CA (:attr:`ca_certificate_path`) read-only
    - inject placeholder credential env values
      (:attr:`placeholder_env`) so in-container tools get a non-empty
      token to attempt their calls with

    The binding also retains :attr:`agent_id` and :attr:`secret_ids`
    so the gateway can issue a ``DELETE /api/agents/{id}`` (and
    matching secret deletions) at shutdown.

    Frozen so the binding is safe to share across the agent's
    sandbox lifetime; the underlying broker secrets do not change
    during a single load.
    """

    agent_id: str
    """OneCLI agent UUID (opaque to us)."""

    secret_ids: tuple[str, ...]
    """OneCLI secret UUIDs registered for this agent."""

    access_token: ServiceCredential
    """Per-agent proxy access token (``aoc_...``).  The sandbox
    container's ``HTTPS_PROXY`` URL embeds this."""

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

    git_extra_headers: tuple[tuple[str, str], ...] = ()
    """``(git_host, "Authorization: Basic <base64(x:<placeholder>)>")``
    entries the sandbox should materialise as ``http.<url>.extraHeader``
    gitconfig lines.

    Each entry arises from a :class:`BrokerCredentialPlan` whose
    :attr:`~thorn.core._brokering.BrokerCredentialPlan.git_extra_header_host`
    is set.  The gateway renders a per-agent ephemeral gitconfig
    from this list and bind-mounts it read-only at
    :data:`~thorn.sandbox._container.CONTAINER_GIT_CONFIG_PATH`; the
    in-sandbox ``git`` then sends the placeholder ``Authorization``
    header on every request to the host, and the broker rewrites it
    to the real credential on the wire.

    Default empty tuple: bindings without any git-extra-header plans
    don't force a gitconfig mount.
    """

    git_config_path: str | None = None
    """Host filesystem path to the per-agent gitconfig file, if any.

    Populated by the gateway after it renders the gitconfig
    implied by :attr:`git_extra_headers`; ``None`` when the binding
    has no git-extra-header entries (no file written, no mount
    needed).  The sandbox runtime uses this to populate
    :attr:`~thorn.sandbox._container.ContainerHostConfig.git_config_host_path`.
    """

    def __repr__(self) -> str:
        """Render a diagnostic-safe binding summary."""
        placeholder_env = tuple(
            (name, REDACTED_SECRET) for name, _value in self.placeholder_env
        )
        git_extra_headers = tuple(
            (host, redact_secrets(header_value))
            for host, header_value in self.git_extra_headers
        )
        return (
            "BrokerBinding("
            f"agent_id={self.agent_id!r}, "
            f"secret_ids={self.secret_ids!r}, "
            f"access_token={self.access_token!r}, "
            f"proxy_url={redact_secrets(self.proxy_url)!r}, "
            f"ca_certificate_path={self.ca_certificate_path!r}, "
            f"placeholder_env={placeholder_env!r}, "
            f"git_extra_headers={git_extra_headers!r}, "
            f"git_config_path={self.git_config_path!r}"
            ")"
        )


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


ServiceLookup = Callable[[str], "Service"]
"""Callable resolving a service name to a registered :class:`Service`.

Typically the gateway passes :meth:`Runtime.get_service` directly.
Defining the alias here keeps the broker module from importing the
``Runtime`` class -- the broker only needs the lookup capability,
not the full runtime.  The callable should raise :class:`KeyError`
when no service with the given name is registered.
"""


def register_agent_with_broker(
    *,
    client: BrokerClient,
    agent: "Agent",
    service_lookup: ServiceLookup,
    ca_certificate_path: str,
) -> BrokerBinding:
    """Register *agent*'s broker-routed credentials with the broker.

    Performs the full agent-load registration sequence:

    1. Walk the agent's accounts.  For each account whose
       backing :class:`Service` implements
       :class:`~thorn.core._brokering.BrokerableService`, ask the
       service for its
       :class:`~thorn.core._brokering.BrokerCredentialPlan`\\ s.
    2. For each plan: read the literal secret value from
       ``os.environ[plan.env_var_name]``, register an OneCLI secret
       carrying the value with the plan's host/path policy and
       injection config, and remember the resulting secret ID.
       Generate a placeholder env entry keyed on the same env var
       name so the sandbox container sees a non-empty value
       in-place.
    3. Create a fresh OneCLI agent and mint its proxy access token.
    4. Bind the registered secrets to the freshly-created agent.

    The agent state itself is not mutated -- credentials never held
    the literal value in the first place; the env var stays in
    ``os.environ`` and the placeholder ends up in the sandbox
    container's environment via the returned :class:`BrokerBinding`.

    Accounts whose service does not implement
    :class:`BrokerableService` are silently skipped (e.g. project
    services, future non-credential service families).

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
    sandbox-launch wiring.
    """
    if agent.id is None:
        raise BrokerError(
            "Cannot register an agent without an id with the broker"
        )

    plans = _collect_plans(agent, service_lookup)

    # Phase 1: register each credential as a secret.  We resolve the
    # literal value from ``os.environ`` here and forward it to the
    # broker.  The literal does NOT enter agent state.
    #
    # A service may return several plans referencing the same env
    # var (e.g. GitHub's forge service ships both an API-routing
    # plan and a git-HTTPS plan for the same PAT).  We want the
    # sandbox to see a single, consistent placeholder per env var
    # so that in-container tools derive the same Basic-auth bytes
    # for the Git extraHeader gitconfig line as they would see in
    # ``GITHUB_TOKEN`` -- otherwise the gitconfig's literal payload
    # wouldn't match what the running container believes its token
    # to be.  ``env_var_to_placeholder`` enforces the sharing.
    secret_ids: list[str] = []
    placeholder_env: list[tuple[str, str]] = []
    env_var_to_placeholder: dict[str, str] = {}
    git_extra_headers: list[tuple[str, str]] = []
    for service_name, plan, credential in plans:
        try:
            value = credential.read_value()
        except CredentialMissingError as exc:
            raise BrokerError(str(exc)) from exc
        registered_value = str(value)
        if plan.value_transform is not None:
            # The transform exists so the service can pre-encode
            # the secret payload to match its injection config
            # (e.g. base64(x:<pat>) for Git HTTPS Basic auth).  We
            # run it exactly once per plan; OneCLI stores the
            # returned string verbatim.
            registered_value = plan.value_transform(registered_value)
        registration = client.register_secret(
            name=_broker_secret_name(agent.id, service_name, plan),
            value=registered_value,
            host_pattern=plan.host_pattern,
            path_pattern=plan.path_pattern,
            injection=plan.injection,
        )
        secret_ids.append(registration.secret_id)
        placeholder = env_var_to_placeholder.get(plan.env_var_name)
        if placeholder is None:
            placeholder = _make_placeholder_value()
            env_var_to_placeholder[plan.env_var_name] = placeholder
            placeholder_env.append((plan.env_var_name, placeholder))

        if plan.git_extra_header_host is not None:
            # The gitconfig's ``extraHeader`` is injected verbatim
            # by ``git`` into the initial request, so it needs to
            # look like a real Basic auth header shape.  The
            # broker overrides the *value* bytes, not the header
            # name, so we pre-compute the placeholder-backed
            # Basic-encoded form here and let OneCLI swap in the
            # real ``base64(x:<pat>)`` value on the wire.  Using
            # the shared placeholder keeps anything in the sandbox
            # that might inspect the gitconfig + GITHUB_TOKEN pair
            # mutually consistent.
            encoded = base64.b64encode(
                f"x:{placeholder}".encode(),
            ).decode()
            header_value = f"Authorization: Basic {encoded}"
            git_extra_headers.append(
                (plan.git_extra_header_host, header_value),
            )

    # Phase 2: create the OneCLI agent and bind the secrets.
    agent_registration = client.register_agent(
        identifier=_broker_identifier_for_agent(agent),
        name=f"thorn:{agent.id}",
    )
    if secret_ids:
        client.bind_secrets_to_agent(
            agent_registration.agent_id, secret_ids,
        )

    proxy_url = client.proxy_url_for_agent(agent_registration.access_token)

    return BrokerBinding(
        agent_id=agent_registration.agent_id,
        secret_ids=tuple(secret_ids),
        access_token=agent_registration.access_token,
        proxy_url=proxy_url,
        ca_certificate_path=ca_certificate_path,
        placeholder_env=tuple(placeholder_env),
        git_extra_headers=tuple(git_extra_headers),
        # Gateway fills this in after writing the gitconfig file
        # to its per-agent sandbox dir; the registration step
        # doesn't know the on-disk location.  See
        # :meth:`Gateway._register_broker_bindings` for the fill.
        git_config_path=None,
    )


def _collect_plans(
    agent: "Agent",
    service_lookup: ServiceLookup,
) -> list[tuple[str, BrokerCredentialPlan, Credential]]:
    """Walk the agent's accounts and collect per-credential plans.

    Returns a list of ``(service_name, plan, credential)`` triples,
    one per credential the broker should register.  Accounts whose
    service does not exist on the runtime raise :class:`BrokerError`
    so misconfigured agents fail fast at registration time rather
    than producing confusing downstream errors.

    Accounts whose service is registered but does not implement
    :class:`BrokerableService` are silently skipped: that is how a
    service tells us "no broker-routed credentials here, please".
    """
    accounts = getattr(agent, "accounts", None)
    if accounts is None:
        return []

    plans: list[tuple[str, BrokerCredentialPlan, Credential]] = []
    for account in accounts.accounts:
        try:
            service = service_lookup(account.service)
        except KeyError as exc:
            raise BrokerError(
                f"Agent {agent.name!r} declares an account on service "
                f"{account.service!r}, which is not registered on "
                "the runtime.  This typically means the service was "
                "not declared in gateway.json (or could not be "
                "inferred from a project URL)."
            ) from exc

        if not isinstance(service, BrokerableService):
            # Service exists but does not participate in broker
            # registration -- silently skip.  Future service
            # families (project services, event sources) live here
            # naturally without the broker module needing to know
            # about them.
            continue

        service_plans = service.broker_credential_plans(account)
        # Pair plans back to the agent's credentials by env var name
        # so the registration loop above can call read_value() on
        # the right Credential instance (and, in turn, surface a
        # clear error if the env var is unset).  The service's plans
        # name an env var; the account's credentials list names env
        # vars; we look up the matching credential here so the
        # broker code never has to reach into per-credential fields
        # itself.
        creds_by_env_var = {c.env_var_name: c for c in account.credentials}
        for plan in service_plans:
            cred = creds_by_env_var.get(plan.env_var_name)
            if cred is None:
                # The service planned for an env var the account
                # doesn't carry -- a wiring bug in the service's
                # plan-construction logic.  Surface loudly.
                raise BrokerError(
                    f"Service {service.name!r} planned a broker "
                    f"registration for env var {plan.env_var_name!r}, "
                    "but the agent's account on that service has no "
                    "credential referencing that env var."
                )
            plans.append((service.name, plan, cred))
    return plans


def _broker_secret_name(
    agent_id: Any,
    service_name: str,
    plan: BrokerCredentialPlan,
) -> str:
    """Compose a stable, operator-readable broker secret name.

    The name surfaces in OneCLI's UI / logs, so we make it
    human-meaningful: ``"<agent_id>-<service>-<plan-suffix>"``.  The
    suffix is service-chosen (see
    :attr:`BrokerCredentialPlan.secret_name_suffix`).
    """
    return f"{agent_id}-{service_name}-{plan.secret_name_suffix}"


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


def admin_api_key_from_env(broker_config: BrokerConfig) -> ServiceCredential:
    """Read the admin API key for an external broker out of ``os.environ``.

    Returns a :class:`ServiceCredential` so the value redacts on
    repr.  Raises :class:`BrokerError` (with a clear message) when
    the env var named by
    :attr:`BrokerConfig.admin_api_key_env_var` is unset or when the
    config does not name an env var.

    Bundled-broker mode mints the key in process memory and supplies
    it directly to the client; this helper is only used by the
    external-broker path.
    """
    name = broker_config.admin_api_key_env_var
    if name is None:
        raise BrokerError(
            "BrokerConfig.admin_api_key_env_var is unset; cannot "
            "resolve the admin API key.  This typically means the "
            "broker block in gateway.json is missing the "
            "admin_api_key_env_var field for an external broker."
        )
    try:
        value = os.environ[name]
    except KeyError as exc:
        raise BrokerError(
            f"Environment variable {name!r} (named by "
            "BrokerConfig.admin_api_key_env_var) is not set; "
            "export the variable in the gateway's environment "
            "and restart `thorn serve`."
        ) from exc
    return ServiceCredential(value)


__all__ = [
    "AgentRegistration",
    "BrokerBinding",
    "BrokerClient",
    "BrokerError",
    "HeaderInjection",
    "InjectionConfig",
    "ParamInjection",
    "SecretRegistration",
    "ServiceLookup",
    "admin_api_key_from_env",
    "register_agent_with_broker",
]
