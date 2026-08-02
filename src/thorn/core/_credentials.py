"""Credential reference and in-process redaction wrapper.

This module defines two types that are deliberately kept narrow:

- :class:`Credential` -- a *reference* to a secret that lives in an
  environment variable.  An agent account's ``credentials`` list is
  built from these.  A ``Credential`` carries no literal secret value;
  it carries the *name* of the env var the operator chose to put the
  secret into.  The value itself is read from ``os.environ`` at the
  point of use (broker registration, direct authentication, etc.) by
  calling :meth:`Credential.read_value`.

- :class:`ServiceCredential` -- a thin :class:`str` subclass whose
  ``__repr__`` redacts the underlying value.  Used as the type for
  *in-process* secret strings (broker access tokens fetched from the
  broker, secret values read out of env vars, etc.) so that printing
  an object that holds one does not leak the secret into logs or
  error messages.

Why not just store ``value: str`` on ``Credential``?

Because Thorn never needs the literal secret to live in agent state.
The broker-registration path reads ``os.environ[env_var_name]`` once,
forwards the value to the credential broker over its TLS-bridged
admin connection, and immediately drops the literal -- only the
broker stores it after that.  Sandbox containers receive a
placeholder env value plus an ``HTTPS_PROXY`` URL pointing at the
broker; the upstream HTTP request is the broker's first sight of
the literal token, and it substitutes the real value into the
matching outbound header before forwarding.

Storing the literal on ``Credential`` would mean every loaded agent
state object on the gateway side held a copy of the literal, which
would defeat the point of having the broker -- a memory dump of the
gateway process should *not* be enough to recover all the agents'
upstream credentials.

Why is the kind a free-form ``str``?

Because the credential ecosystem is not yet settled.  We only have
two concrete kinds today (``"pat"`` for GitHub PATs / generic
bearer tokens, ``"gitlab-pat"`` for GitLab personal access tokens),
and the structure of e.g. GitHub App auth is still under design.
Per-service code matches on ``kind`` to decide what to do; there is
no central registry that would have to be updated when a service
introduces a new credential kind.  When the design firms up we can
revisit and introduce a typed enum / discriminated-union here.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class ServiceCredential(str):
    """A redacted-on-repr string subclass for in-process secret values.

    Subclass of :class:`str`, so any function that takes a ``str``
    Just Works: ``cred`` is the credential string, ``str(cred)`` is
    the same string, equality and hashing match plain ``str``.  The
    only behavior that differs from ``str`` is :meth:`__repr__`,
    which redacts the underlying value entirely.

    This type intentionally carries no metadata (no ``state`` field,
    no audit hooks).  Its sole job is logging hygiene -- it exists
    so that a stray ``log.info("got %r", cred)`` or a
    ``ValidationError`` rendering an object that holds a credential
    cannot leak the secret into operator-visible logs.

    ``Credential.read_value`` returns one of these; broker-side code
    that mints fresh access tokens wraps them in this type before
    handing them off to anything that might log the value.
    """

    def redacted(self) -> str:
        """Return a logging-safe summary that hides the underlying value."""
        return f"<redacted len={len(self)}>"

    def __repr__(self) -> str:
        return f"ServiceCredential({self.redacted()})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str.__str__,
                return_schema=core_schema.str_schema(),
                when_used="always",
            ),
        )

    @classmethod
    def _validate(cls, value: Any) -> "ServiceCredential":
        if isinstance(value, ServiceCredential):
            return value
        if isinstance(value, str):
            return cls(value)
        raise TypeError(
            f"ServiceCredential requires str input, "
            f"got {type(value).__name__}"
        )


class CredentialMissingError(LookupError):
    """Raised when a :class:`Credential`'s env var is not set in os.environ.

    Distinct from :class:`KeyError` so callers can distinguish "the
    operator never set the secret" from incidental dict misses, and
    so the error message can mention the credential's logical
    identity (kind + env var name) rather than just the env var key.
    """


class Credential(BaseModel):
    """A reference to a secret stored in an environment variable.

    *kind* is a free-form short string identifying what kind of
    credential this is from the consuming service's point of view
    (e.g. ``"pat"``, ``"gitlab-pat"``).  Service-side code matches
    on this to decide how to use the credential.  Operators write the
    kind in their agent JSON files and Thorn does not validate the
    string against any registry -- a typo means the consuming service
    will not recognise the credential and surface an error of its
    own.

    *name* is an optional human-facing label (``"primary"``,
    ``"backup"``, etc.) used to disambiguate multiple credentials of
    the same kind on the same account.  Most accounts have exactly
    one credential and leave this unset.

    *env_var_name* is the name of the environment variable the
    operator put the literal secret into.  Thorn reads that env var
    *only* at the points where it is genuinely needed -- broker
    registration, direct service authentication when the broker is
    not in use -- via :meth:`read_value`.  Persisted agent state
    never holds the literal value; only the env var name (which is
    not itself a secret).
    """

    model_config = ConfigDict(frozen=True)

    kind: str = Field(
        min_length=1,
        description=(
            "Short identifier for this credential's shape from the "
            "consuming service's point of view (e.g. 'pat')."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Optional human-facing label disambiguating multiple "
            "credentials of the same kind on a single account."
        ),
    )
    env_var_name: str = Field(
        min_length=1,
        description=(
            "Name of the environment variable holding the literal "
            "secret.  Resolved via ``os.environ`` at use time; not "
            "stored anywhere in agent state."
        ),
    )

    def read_value(self) -> ServiceCredential:
        """Read the literal credential value from ``os.environ``.

        Returns a :class:`ServiceCredential` so that any logging
        surface that incidentally holds the value redacts on
        ``repr``.

        Raises :class:`CredentialMissingError` (a ``LookupError``
        subclass) when the env var is not set; callers are expected
        to surface that to the operator so they can set the variable
        and retry.
        """
        try:
            raw = os.environ[self.env_var_name]
        except KeyError as exc:
            raise CredentialMissingError(
                f"Credential (kind={self.kind!r}, "
                f"name={self.name!r}) references environment variable "
                f"{self.env_var_name!r}, which is not set.  Export "
                f"the variable in the gateway's environment and "
                "restart `thorn serve`."
            ) from exc
        return ServiceCredential(raw)

    def __repr__(self) -> str:
        return (
            f"Credential(kind={self.kind!r}, name={self.name!r}, "
            f"env_var_name={self.env_var_name!r})"
        )


__all__ = [
    "Credential",
    "CredentialMissingError",
    "ServiceCredential",
]
