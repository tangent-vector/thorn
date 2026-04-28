"""Service credential wrapper distinguishing literal vs placeholder values.

A :class:`ServiceCredential` is a string-subclass wrapper around a value
that holds (or held) authentication material for an external service.
The wrapper carries a ``state`` (``"literal"`` or ``"placeholder"``) so
that gateway code, logs, and audit tests can tell at a glance whether a
particular credential instance currently carries a real secret or a
broker placeholder.

Two reasons this type exists:

- **Audit invariant.**  Phase D's invariant is "post-registration, no
  ``ServiceCredential`` reachable from the gateway's loaded agent state
  carries a literal value -- everything has been swapped to a broker
  placeholder."  Without an explicit type, that invariant cannot be
  expressed mechanically; with one, the audit reduces to a tree walk
  via :func:`walk_credentials`.

- **Logging hygiene.**  ``__repr__`` redacts the value entirely so that
  printing an agent's loaded state, formatting a ``ValidationError``,
  or dumping a Pydantic model never leaks live tokens.

``ServiceCredential`` subclasses :class:`str` so existing call sites
that pass the value to HTTP clients, git commands, and the like do not
break: ``cred`` is the credential string, ``str(cred)`` is the same
string, equality and hashing match plain ``str``.  Code that wants to
*test* the state should use :attr:`ServiceCredential.is_placeholder` /
:attr:`is_literal` or match on :attr:`state`; code that wants to
*display* the credential should call :func:`repr` (or
:meth:`redacted`), which will not include the underlying value.

Pydantic v2 integration is via :meth:`__get_pydantic_core_schema__`:
plain strings coming from JSON / dict input are wrapped as
``"literal"`` credentials by default (matching the existing
``$ENV_VAR`` -> ``str`` path in gateway config loading); already-wrapped
values are passed through unchanged so re-validation preserves state.
Serialization uses the underlying string, which means
``model_dump_json()`` of a placeholder model produces a JSON string
that, on re-load, comes back as ``"literal"`` -- this is intentional:
placeholder state lives only in-process, never on disk.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

CredentialState = Literal["literal", "placeholder"]
"""Lifecycle state of a :class:`ServiceCredential` instance.

- ``"literal"``: the wrapped value is a real credential read from the
  operator's configuration (env var, agent.json, gateway.json).  Found
  in agent state during the agent-load window before broker
  registration runs.
- ``"placeholder"``: the wrapped value is a broker-issued placeholder
  string.  Real credential lives in the broker; this string only needs
  to be non-empty so in-container HTTP clients attempt the call.
"""


class ServiceCredential(str):
    """A token / API key / similar secret with explicit lifecycle state.

    Subclass of :class:`str`; passing this to any function that takes a
    ``str`` Just Works.  The :attr:`state` attribute records whether
    the wrapped value is a real credential (``"literal"``) or a broker
    placeholder (``"placeholder"``).

    Construct with the keyword-only ``state`` argument::

        ServiceCredential("ghp_abc...", state="literal")
        ServiceCredential("aoc_placeholder", state="placeholder")

    Pydantic v2 validates plain strings into ``"literal"``-state
    instances by default; the broker registration code is the (only)
    place that constructs ``"placeholder"``-state instances.
    """

    state: CredentialState

    def __new__(
        cls,
        value: str,
        *,
        state: CredentialState = "literal",
    ) -> "ServiceCredential":
        if state not in ("literal", "placeholder"):
            raise ValueError(
                f"invalid credential state {state!r}: "
                "must be 'literal' or 'placeholder'"
            )
        # Empty values are allowed: forge service-level configs use
        # ``ServiceCredential("")`` as a structural placeholder when
        # the real credential comes from a per-agent
        # :class:`ForgeAccountConfig` at call time.  The audit
        # invariant (:func:`assert_no_literal_credentials`) treats
        # empty literal credentials as harmless because they cannot
        # carry meaningful auth material.
        instance = super().__new__(cls, value)
        # str subclasses don't support __dict__; we reach around with
        # object.__setattr__ to attach state without enabling general
        # mutability.
        object.__setattr__(instance, "state", state)
        return instance

    @property
    def is_placeholder(self) -> bool:
        """``True`` iff this credential is a broker placeholder."""
        return self.state == "placeholder"

    @property
    def is_literal(self) -> bool:
        """``True`` iff this credential carries a real secret."""
        return self.state == "literal"

    def redacted(self) -> str:
        """Return a logging-safe summary that hides the underlying value."""
        return f"<{self.state} len={len(self)}>"

    def with_state(self, state: CredentialState) -> "ServiceCredential":
        """Return a fresh ``ServiceCredential`` wrapping the same value but
        in *state*.

        Used by registration code that wants to swap a literal value
        for a placeholder without mutating the existing instance.
        """
        return ServiceCredential(str.__str__(self), state=state)

    def __repr__(self) -> str:
        return f"ServiceCredential({self.redacted()})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        # Use a plain validator (not after-str-validator) so an
        # already-wrapped ``ServiceCredential`` passes through with its
        # state intact.  An after-validator would let Pydantic's
        # ``str_schema`` coerce the subclass back to plain ``str``
        # before our wrapper runs, silently downgrading any
        # ``placeholder``-state credential to ``literal``.
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                return_schema=core_schema.str_schema(),
                when_used="always",
            ),
        )

    @classmethod
    def _validate(cls, value: Any) -> "ServiceCredential":
        if isinstance(value, ServiceCredential):
            return value
        if isinstance(value, str):
            return cls(value, state="literal")
        raise TypeError(
            f"ServiceCredential requires str input, "
            f"got {type(value).__name__}"
        )

    @staticmethod
    def _serialize(value: "ServiceCredential") -> str:
        return str.__str__(value)


# ---------------------------------------------------------------------------
# Audit traversal
# ---------------------------------------------------------------------------


def walk_credentials(obj: Any) -> Iterator[ServiceCredential]:
    """Yield every :class:`ServiceCredential` reachable from *obj*.

    Walks Pydantic models (via ``model_fields``), plain objects (via
    ``__dict__``), dicts, and iterable sequences.  Used by the Phase D
    audit assertion to confirm that no literal-state credential
    survives in a loaded gateway state after broker registration.

    Identity-based deduplication via :func:`id` prevents infinite loops
    on cyclic graphs and avoids visiting interned strings repeatedly.
    """
    seen: set[int] = set()
    stack: list[Any] = [obj]
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)

        if isinstance(node, ServiceCredential):
            yield node
            continue

        # Leaf scalar types: nothing to descend into.  ``str`` matches
        # plain strings (after the ServiceCredential check above).
        if isinstance(node, (str, bytes, int, float, bool)) or node is None:
            continue

        if isinstance(node, dict):
            stack.extend(node.values())
            continue

        if isinstance(node, (list, tuple, set, frozenset)):
            stack.extend(node)
            continue

        # Pydantic v2 BaseModel exposes its fields via ``model_fields``
        # on the class (instance access is deprecated in Pydantic
        # v2.11+).  Iterate by field name so we use the validated
        # attribute access path rather than internal storage.
        node_cls = type(node)
        cls_model_fields = getattr(node_cls, "model_fields", None)
        if cls_model_fields is not None:
            for field_name in cls_model_fields:
                stack.append(getattr(node, field_name, None))
            continue

        if hasattr(node, "__dict__"):
            stack.extend(node.__dict__.values())


def assert_no_literal_credentials(obj: Any) -> None:
    """Raise :class:`AssertionError` if any non-empty literal-state
    credential is reachable from *obj*.

    The audit invariant for Phase D's broker integration:
    post-registration, every meaningful ``ServiceCredential`` instance
    reachable from the loaded gateway state must be in ``placeholder``
    state.  Tests and (eventually) operator-facing diagnostics use
    this helper to enforce that invariant.

    Empty literal credentials are tolerated because they are
    structural shims (forge service-level configs use ``""`` as a
    "no service-level credential, fill from per-agent account at
    call time" sentinel) and carry no auth material that could leak.
    """
    offenders = [
        c for c in walk_credentials(obj) if c.is_literal and len(c) > 0
    ]
    if offenders:
        # Don't include the values themselves -- this is the whole
        # point of the redaction story.
        summaries = ", ".join(c.redacted() for c in offenders)
        raise AssertionError(
            f"audit failure: {len(offenders)} literal-state credential(s) "
            f"reachable from object: {summaries}"
        )


__all__ = [
    "CredentialState",
    "ServiceCredential",
    "assert_no_literal_credentials",
    "walk_credentials",
]
