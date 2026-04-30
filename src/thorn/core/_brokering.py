"""Service-side credential brokering protocol.

This module defines the ABCs and value types that let a
:class:`~thorn.core.Service` describe -- without the broker module
needing to know anything about the service's specifics -- how its
credentials should be registered with an upstream HTTP credential
broker (today: OneCLI).

Why a service-side protocol?

The broker code itself (:mod:`thorn.gateway._broker`) is
deliberately ignorant of which services exist, what credential
shapes they accept, what host/path patterns their upstream APIs use,
or what header name carries their authorization.  All of that is
service-specific knowledge that belongs in the service module.

The broker code's job is mechanical: walk an agent's accounts, ask
each account's service "do you participate in broker registration,
and if so what plans do you have?", and -- for each plan -- read
the literal secret out of ``os.environ[plan.env_var_name]`` and
register it with the broker.  The plans the service returns describe
the host/path policy and injection config the broker needs to
configure substitution; they do *not* carry the literal secret.

Adding a new service that needs broker integration is therefore a
purely local change: subclass :class:`BrokerableService`, implement
:meth:`broker_credential_plans`, and you're done -- no edits to the
broker module required.

Injection config types
----------------------

OneCLI's substitution model supports two shapes today: header
injection (set/override an HTTP header on outbound requests
matching the host/path policy) and query-parameter injection
(append a URL query parameter).  We model both as small Pydantic
classes so service code constructs them with named arguments and
the broker driver translates them to OneCLI's wire shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, Field

from thorn.core._account import AccountConfig
from thorn.core._service import Service


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


@dataclass(frozen=True)
class BrokerCredentialPlan:
    """A service's recipe for registering one credential with the broker.

    Returned by :meth:`BrokerableService.broker_credential_plans` --
    one plan per credential the service wants registered.  The
    broker driver consumes these to:

    1. Read the literal secret value from
       ``os.environ[env_var_name]``.
    2. Register a broker secret with the given host/path policy and
       injection config.
    3. Compute a per-credential placeholder env entry for the
       sandbox container's environment, keyed on the same
       ``env_var_name`` (so the in-container tool sees a non-empty
       value to attempt the call with, while the broker substitutes
       the real value into matching outbound requests).

    Frozen so plans are safe to share across the agent's broker
    registration; nothing about the plan changes after it is
    constructed.

    *secret_name_suffix* is appended (after a hyphen) to a unique
    per-(agent, account) prefix the broker driver chooses, giving
    a stable, human-readable broker secret name without the service
    having to know the full naming convention.
    """

    env_var_name: str
    """Name of the env var holding the literal secret.  The broker
    driver reads this from ``os.environ`` to obtain the value, and
    uses the same name to inject the placeholder env entry into the
    sandbox container so in-container tools find a non-empty value
    under the env var they expect."""

    host_pattern: str
    """OneCLI ``hostPattern`` -- the bare hostname matching outbound
    requests this credential authenticates.  No scheme, no path."""

    path_pattern: str | None
    """OneCLI ``pathPattern`` (glob).  ``None`` means "all paths under
    the host"; passing ``"/api/*"`` (etc.) narrows the match."""

    injection: InjectionConfig
    """How the broker should splice the secret into matching
    outbound requests."""

    secret_name_suffix: str
    """Service-chosen suffix appended to the broker driver's
    per-(agent, account) prefix to form the broker secret's
    operator-visible name.  Should be short and identify the
    credential's role (e.g. ``"github-pat"``)."""


class BrokerableService(Service, ABC):
    """A service that participates in upstream credential brokering.

    Mixin-style ABC: services that authenticate to upstream HTTPS
    APIs and want their credentials proxied through OneCLI subclass
    this *in addition to* whatever else they subclass (e.g. a forge
    host service).  The broker code uses an ``isinstance``-check
    against this class to decide whether an account's service
    participates in registration; services that don't participate
    are silently skipped.

    Implementers fill in :meth:`broker_credential_plans`.  Everything
    else (reading the literal value from ``os.environ``, registering
    with OneCLI, building placeholder env entries) is the broker
    driver's job.
    """

    @abstractmethod
    def broker_credential_plans(
        self,
        account: AccountConfig,
    ) -> list[BrokerCredentialPlan]:
        """Return one plan per credential to register with the broker.

        *account* is the typed (validation-pass output) account
        instance for this service.  Implementers should
        ``isinstance``-check it against the service's declared
        :attr:`Service.AccountConfig` subclass and raise
        :class:`TypeError` on mismatch -- the gateway's validation
        pass should make a mismatch impossible, but the explicit
        check turns wiring bugs into clear errors at the failing
        site.

        Return an empty list when the account has no credentials
        the broker can handle (e.g. an account that uses a
        credential kind this service does not yet broker-route).
        Returning an empty list is *not* an error -- the agent will
        load with no broker registration for that account.
        """


__all__ = [
    "BrokerableService",
    "BrokerCredentialPlan",
    "HeaderInjection",
    "InjectionConfig",
    "ParamInjection",
]
