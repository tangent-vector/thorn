"""Base class for named entities in an agency.

A ``Service`` is any named entity declared in the agency configuration
(e.g. ``gateway.json``).  Forge connections, project definitions, and
event sources are all services.  The :class:`~thorn.runtime.Runtime`
hosts a registry of services and provides lookup by name or type.

All concrete ``Service`` subclasses must define a ``Config`` class
attribute (a :class:`pydantic.BaseModel`) describing their
configuration schema, and accept an instance of that model as the
sole positional constructor argument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pydantic import BaseModel


class Service(ABC):
    """Base class for named entities in an agency.

    Subclasses must:

    1. Define a ``Config`` class attribute pointing to a
       :class:`pydantic.BaseModel` subclass.
    2. Accept a ``Config`` instance as the sole positional argument
       to ``__init__``.
    3. Implement the :attr:`name` property.
    """

    Config: ClassVar[type[BaseModel]]

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this service within the agency."""


__all__ = [
    "Service",
]
