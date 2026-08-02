"""Base class for named entities in an agency.

A ``Service`` is any named entity declared in the agency configuration
(e.g. ``gateway.json``).  Forge connections, project definitions, and
event sources are all services.  The :class:`~thorn.runtime.Runtime`
hosts a registry of services and provides lookup by name or type.

All concrete ``Service`` subclasses must define a ``Config`` class
attribute (a :class:`pydantic.BaseModel`) describing their
configuration schema, and accept an instance of that model as the
sole positional constructor argument.

Account support
---------------

A ``Service`` subclass that supports per-agent accounts (i.e. wants
to be referenced from an agent's ``accounts[]``) sets the
:attr:`AccountConfig` class attribute to a
:class:`~thorn.core._account.AccountConfig` subclass describing the
account shape it expects.  The gateway's startup pass uses
:meth:`validate_account` to convert each agent's
:class:`~thorn.core._account.UntypedAccountConfig` parse-time entry
into the typed shape declared here.

Services that don't support accounts (project services, event
sources, etc.) leave :attr:`AccountConfig` set to ``None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pydantic import BaseModel

    from thorn.core._account import AccountConfig, UntypedAccountConfig


class Service(ABC):
    """Base class for named entities in an agency.

    Subclasses must:

    1. Define a ``Config`` class attribute pointing to a
       :class:`pydantic.BaseModel` subclass.
    2. Accept a ``Config`` instance as the sole positional argument
       to ``__init__``.
    3. Implement the :attr:`name` property.
    """

    Config: ClassVar[type["BaseModel"]]
    AccountConfig: ClassVar[type["AccountConfig"] | None] = None
    """Pydantic model describing this service's per-agent account shape.

    Set by services that support being referenced from an agent's
    ``accounts[]``.  Left ``None`` for services that don't (project
    services, event sources, etc.).  The gateway's account-validation
    pass uses this to convert untyped parse-time account entries into
    typed instances at startup.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this service within the agency."""

    def validate_account(
        self,
        raw: "UntypedAccountConfig",
    ) -> "AccountConfig":
        """Validate one parse-time account entry against this service's shape.

        Default implementation routes through the
        :attr:`AccountConfig` ClassVar: if it is set, the raw entry's
        fields (including the ``extra``-allowed per-service fields)
        are re-validated against that model and returned.  If it is
        not set, the service has declared that it does not support
        accounts and a :class:`TypeError` is raised so the operator
        sees a clear error rather than a silent no-op.

        Services with non-trivial validation (e.g. picking a concrete
        subclass based on the credentials list) can override this
        method.
        """
        from thorn.core._account import AccountConfig

        account_config_cls = type(self).AccountConfig
        if account_config_cls is None:
            raise TypeError(
                f"Service {self.name!r} (class "
                f"{type(self).__name__}) does not support accounts; "
                "an agent declared an account on it."
            )
        if not issubclass(account_config_cls, AccountConfig):
            raise TypeError(
                f"Service {type(self).__name__}.AccountConfig must "
                f"subclass thorn.core._account.AccountConfig; got "
                f"{account_config_cls.__name__}."
            )

        # Round-trip through model_dump so per-service ``extra``
        # fields parsed by the untyped model are visible to the
        # typed model's validator.
        return account_config_cls.model_validate(raw.model_dump())


__all__ = [
    "Service",
]
