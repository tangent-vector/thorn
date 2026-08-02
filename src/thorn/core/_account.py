"""Agent account models -- generic shape and parse-time untyped form.

An agent's identity on an external service is represented by an
account entry under the ``"accounts"`` key of its identity JSON file.
The on-disk shape is a flat list whose entries each carry a
``service`` discriminator plus a ``credentials`` list and any
service-specific fields::

    "accounts": [
      { "service": "github",
        "git_user_name": "thorn-agent",
        "git_user_email": "thorn@example.com",
        "credentials": [
          {"kind": "pat", "env_var_name": "GITHUB_TOKEN"}
        ] }
    ]

Two-stage typing
----------------

The framework deliberately parses account entries in two stages:

1. **Parse-time (untyped).**  :class:`UntypedAccountConfig` holds
   the ``service`` field and the credentials list, but accepts
   arbitrary additional keys (``model_config = extra="allow"``) and
   does not validate any per-service fields.  This is the shape the
   agent JSON deserializer produces, and it is intentionally
   permissive -- at parse time we do not yet know which service the
   ``service`` field refers to or what shape its account config
   takes.

2. **Validation-time (typed).**  Each :class:`~thorn.core.Service`
   subclass that supports accounts declares an ``AccountConfig``
   :class:`pydantic.BaseModel` subclass via a ``ClassVar`` on the
   service class, and the gateway's startup pass walks every loaded
   agent's accounts and replaces each ``UntypedAccountConfig`` with
   the typed concrete result of calling
   :meth:`Service.validate_account` for the corresponding service.

This split is what lets the framework support service-specific
account shapes (e.g. ``git_user_name`` / ``git_user_email`` on a
forge account, hypothetical ``inbox_filter`` on a future email
account) without having a central registry that has to know about
every service ahead of time -- the service module owns its own
account shape.

Resolving accounts at runtime
-----------------------------

The :func:`resolve_account` helper finds an agent's account on a
named service.  It is the primary entry point for code that needs
service-specific fields (e.g. ``git_user_email``) at runtime; for
credential-bearing services that participate in broker
registration, see :class:`thorn.core._brokering.BrokerableService`
instead, which the broker code drives through.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from thorn.core._credentials import Credential

if TYPE_CHECKING:
    from thorn.core._agent import Agent
    from thorn.core._service import Service


ServiceLookup = Callable[[str], "Service"]
"""Callable resolving a service name to a registered :class:`Service`.

Used by :func:`validate_agent_accounts` to look up the service that
each account references, so the service can validate the account
against its declared :class:`AccountConfig` shape.  The callable
should raise :class:`KeyError` when no service with the given name
is registered (matching :meth:`Runtime.get_service`'s contract).
"""


class AccountConfig(BaseModel):
    """Validated, typed account configuration -- base for service subclasses.

    Concrete services subclass this model to add service-specific
    fields (e.g. forge accounts add ``git_user_name`` /
    ``git_user_email``).  All concrete subclasses inherit the
    ``service`` and ``credentials`` fields unchanged.

    Subclasses are wired to their service via
    :attr:`Service.AccountConfig`; the gateway's validation pass
    looks up the service by name and uses its declared class to
    validate each :class:`UntypedAccountConfig` parse-time entry
    against it.
    """

    service: str = Field(
        min_length=1,
        description=(
            "Name of the service this account belongs to.  Must "
            "match the ``name`` of a service registered on the "
            "runtime."
        ),
    )
    credentials: list[Credential] = Field(
        default_factory=list,
        description=(
            "References to the secrets needed to authenticate to "
            "the named service.  Each entry names an env var the "
            "operator put the literal secret into; the value is "
            "read from ``os.environ`` only at the points where it "
            "is needed (broker registration, direct authentication)."
        ),
    )


class UntypedAccountConfig(AccountConfig):
    """Parse-time shape for one ``accounts[]`` entry, before validation.

    Subclass of :class:`AccountConfig` with ``extra='allow'`` so that
    per-service fields (whose shape we don't know until we look up
    the corresponding service) survive parsing intact for the
    gateway's validation pass to consume.

    The deserializer produces a list of these; the gateway's startup
    pass replaces each one with a typed :class:`AccountConfig`
    subclass via :meth:`Service.validate_account`.  ``isinstance(x,
    AccountConfig)`` is True for both shapes, so consumers that only
    need the base fields can be agnostic; consumers that need
    per-service fields ``isinstance``-check against the concrete
    typed subclass.
    """

    model_config = ConfigDict(extra="allow")


class AgentAccountsConfig(BaseModel):
    """All accounts declared for an agent.

    Parsed from the ``"accounts"`` key in an agent's ``.json`` file.
    The list initially holds :class:`UntypedAccountConfig` instances
    (subclasses of :class:`AccountConfig` with ``extra='allow'``);
    after the gateway's validation pass, each entry is replaced with
    the corresponding typed :class:`AccountConfig` subclass.

    The field type is :class:`AccountConfig` so both shapes flow
    through uniformly: at parse time Pydantic validates each entry
    as an :class:`UntypedAccountConfig` (because that's the type
    the deserializer constructs), and the gateway's validation pass
    swaps in the typed subclasses without disturbing the field
    type.  ``model_dump`` recurses through whichever concrete shape
    each entry actually is.
    """

    # Default validator type: UntypedAccountConfig.  We override this
    # in the deserializer by passing pre-built UntypedAccountConfig
    # instances; Pydantic's default model_validate route flows
    # entries that are dict-shaped through UntypedAccountConfig
    # validation, which is exactly the parse-time shape we want.
    accounts: list[AccountConfig] = Field(default_factory=list)


def resolve_account(agent: "Agent", service_name: str) -> AccountConfig:
    """Find the agent's account for the named service.

    Returns the matching account from ``agent.accounts.accounts``.
    Raises :class:`KeyError` (with a descriptive message) when no
    matching account exists, or :class:`TypeError` when the agent's
    account list still holds untyped entries (the gateway's
    validation pass has not run, which is a wiring bug).

    Callers that want to act on service-specific fields should
    ``isinstance``-check the returned account against the concrete
    :class:`AccountConfig` subclass declared by the service they're
    talking to.
    """
    accounts: AgentAccountsConfig | None = getattr(agent, "accounts", None)
    if accounts is None:
        raise KeyError(
            f"Agent {agent.name!r} has no accounts configured.  "
            f"Cannot resolve credentials for service "
            f"{service_name!r}."
        )

    for acct in accounts.accounts:
        if acct.service == service_name:
            if isinstance(acct, UntypedAccountConfig):
                raise TypeError(
                    f"Agent {agent.name!r}'s account on service "
                    f"{service_name!r} is still an "
                    f"UntypedAccountConfig (got {type(acct).__name__}); "
                    "the gateway's per-service validation pass has "
                    "not run yet.  This is a Thorn-internal wiring "
                    "bug; please file an issue."
                )
            return acct

    registered = [a.service for a in accounts.accounts] or ["(none)"]
    raise KeyError(
        f"Agent {agent.name!r} has no account on service "
        f"{service_name!r}.  Configured accounts: {', '.join(registered)}"
    )


def validate_agent_accounts(
    agent: "Agent",
    service_lookup: ServiceLookup,
) -> None:
    """Replace untyped account entries on *agent* with typed ones.

    For each entry in ``agent.accounts.accounts`` whose service is
    registered on the runtime, call
    :meth:`Service.validate_account` and substitute the typed
    result into the list (in place).  Entries whose service is not
    registered are left as ``UntypedAccountConfig`` and a clear
    :class:`ValueError` is raised so the operator sees the
    misconfiguration at startup rather than at first credential
    use.

    Idempotent: entries that are already typed (e.g. when the
    function is called twice) are passed through to
    :meth:`Service.validate_account` again, which produces an
    equivalent typed instance.

    Skips agents that carry no ``accounts`` attribute (e.g. agents
    created in tests without an identity file).
    """
    accounts = getattr(agent, "accounts", None)
    if accounts is None or not accounts.accounts:
        return

    typed_entries: list[AccountConfig] = []
    for entry in accounts.accounts:
        service_name = entry.service
        try:
            service = service_lookup(service_name)
        except KeyError as exc:
            raise ValueError(
                f"Agent {agent.name!r} declares an account on service "
                f"{service_name!r}, which is not registered on the "
                "runtime.  The service must be declared in "
                "gateway.json (or inferable from a project URL) "
                "before the agent loads."
            ) from exc

        # Re-typing untyped entries via the service is the whole
        # point of this pass; we additionally re-validate already-
        # typed entries so callers can run this idempotently
        # (e.g. tests reloading state).  The
        # :meth:`Service.validate_account` default round-trips
        # through ``model_dump`` so per-service ``extra`` fields
        # carried by an UntypedAccountConfig flow through to the
        # typed model's validator.
        from thorn.core._account import UntypedAccountConfig

        if isinstance(entry, AccountConfig) and not isinstance(
            entry, UntypedAccountConfig,
        ):
            # Already typed; preserve as-is rather than re-validating
            # (re-validation would not be wrong, but it does
            # unnecessary work and would constrain future per-service
            # validate_account overrides).
            typed_entries.append(entry)
            continue

        typed_entries.append(service.validate_account(entry))

    accounts.accounts = typed_entries


def find_credential(
    account: AccountConfig | UntypedAccountConfig,
    *,
    kind: str,
    name: str | None = None,
) -> Credential | None:
    """Return the first credential on *account* matching *kind* (and *name*).

    Helper for service-side code that knows which credential kind
    (and optionally name) it wants to consume from an account's
    credentials list.  Returns ``None`` when nothing matches; raising
    is the caller's choice (some sites want to fall through, others
    want to surface an error).

    *name* is matched only when non-``None``; passing ``None`` (the
    default) matches the first credential of *kind* regardless of
    name.
    """
    for cred in account.credentials:
        if cred.kind != kind:
            continue
        if name is not None and cred.name != name:
            continue
        return cred
    return None


def require_credential(
    account: AccountConfig | UntypedAccountConfig,
    *,
    kind: str,
    name: str | None = None,
) -> Credential:
    """Like :func:`find_credential` but raises when no match is found.

    Convenience for callers that have already decided "this account
    must have a credential of this shape" and want a single line for
    the lookup.
    """
    cred = find_credential(account, kind=kind, name=name)
    if cred is None:
        suffix = f" (name={name!r})" if name is not None else ""
        kinds = sorted({c.kind for c in account.credentials}) or ["(none)"]
        raise KeyError(
            f"Account on service {account.service!r} has no "
            f"credential of kind {kind!r}{suffix}.  Available "
            f"kinds: {', '.join(kinds)}"
        )
    return cred


__all__ = [
    "AccountConfig",
    "AgentAccountsConfig",
    "ServiceLookup",
    "UntypedAccountConfig",
    "find_credential",
    "require_credential",
    "resolve_account",
    "validate_agent_accounts",
]
