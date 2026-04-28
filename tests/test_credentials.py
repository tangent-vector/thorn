"""Tests for the ``ServiceCredential`` newtype and its audit helpers.

Covers the contract documented in
:mod:`thorn.core._credentials`: state-tagged str subclassing, repr
redaction, Pydantic v2 round-tripping, the ``walk_credentials``
traversal, and the ``assert_no_literal_credentials`` audit assertion.
"""

from __future__ import annotations

from typing import Annotated, Union

import pytest
from pydantic import BaseModel, Field

from thorn.core._credentials import (
    ServiceCredential,
    assert_no_literal_credentials,
    walk_credentials,
)


# ---------------------------------------------------------------------------
# Construction and state semantics
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_state_is_literal(self):
        cred = ServiceCredential("ghp_abc")
        assert cred.state == "literal"
        assert cred.is_literal
        assert not cred.is_placeholder

    def test_explicit_placeholder(self):
        cred = ServiceCredential("placeholder-xyz", state="placeholder")
        assert cred.state == "placeholder"
        assert cred.is_placeholder
        assert not cred.is_literal

    def test_rejects_invalid_state(self):
        with pytest.raises(ValueError, match="invalid credential state"):
            ServiceCredential("x", state="bogus")  # type: ignore[arg-type]

    def test_allows_empty_value_as_structural_placeholder(self):
        # Empty values exist for forge service-level configs that use
        # ``ServiceCredential("")`` as a "no service-level credential,
        # filled per-agent at call time" sentinel.  Construction
        # succeeds; the audit will skip these.
        empty = ServiceCredential("")
        assert empty == ""
        assert empty.is_literal

    def test_with_state_returns_new_instance(self):
        original = ServiceCredential("real-token", state="literal")
        swapped = original.with_state("placeholder")
        # Underlying value preserved.
        assert str.__str__(swapped) == "real-token"
        assert swapped.is_placeholder
        # Original untouched.
        assert original.is_literal
        assert swapped is not original


# ---------------------------------------------------------------------------
# str compatibility
# ---------------------------------------------------------------------------


class TestStrCompatibility:
    def test_equals_plain_str(self):
        cred = ServiceCredential("ghp_abc")
        assert cred == "ghp_abc"
        assert "ghp_abc" == cred

    def test_hashes_like_str(self):
        cred = ServiceCredential("ghp_abc")
        assert hash(cred) == hash("ghp_abc")

    def test_usable_as_dict_key(self):
        cred = ServiceCredential("ghp_abc")
        d = {cred: 1}
        assert d["ghp_abc"] == 1

    def test_state_does_not_affect_equality(self):
        # Two credentials with the same value but different state are
        # equal as strings -- this is by design (call sites that pass
        # the value around shouldn't accidentally diverge based on
        # state) and is why ``state`` lives on the instance, not in
        # ``__eq__``.
        a = ServiceCredential("token", state="literal")
        b = ServiceCredential("token", state="placeholder")
        assert a == b

    def test_str_concatenation_returns_plain_str(self):
        cred = ServiceCredential("token")
        # Concatenation is a plain str; the cred wrapper does not
        # propagate.  This is fine -- we only need the wrapper to live
        # at the top of agent state, not on every transitively-derived
        # string.
        result = "Bearer " + cred
        assert isinstance(result, str)
        assert result == "Bearer token"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_repr_does_not_leak_value(self):
        cred = ServiceCredential("super-secret-token-xyz")
        text = repr(cred)
        assert "super-secret-token" not in text
        assert "literal" in text
        assert "len=" in text

    def test_repr_distinguishes_states(self):
        literal = ServiceCredential("x" * 32, state="literal")
        placeholder = ServiceCredential("y" * 32, state="placeholder")
        assert "literal" in repr(literal)
        assert "placeholder" in repr(placeholder)

    def test_redacted_method(self):
        cred = ServiceCredential("abc", state="placeholder")
        assert "abc" not in cred.redacted()
        assert cred.redacted() == "<placeholder len=3>"

    def test_str_does_leak_value_intentionally(self):
        # ``str(cred)`` is the underlying value, by design -- call
        # sites pass the credential to HTTP clients using ``str()`` or
        # implicit conversion.  Only ``repr()`` redacts.
        cred = ServiceCredential("ghp_abc")
        assert str(cred) == "ghp_abc"


# ---------------------------------------------------------------------------
# Pydantic integration
# ---------------------------------------------------------------------------


class _CredentialModel(BaseModel):
    token: ServiceCredential


class TestPydanticIntegration:
    def test_validate_from_plain_str(self):
        model = _CredentialModel.model_validate({"token": "ghp_abc"})
        assert isinstance(model.token, ServiceCredential)
        assert model.token.is_literal
        assert model.token == "ghp_abc"

    def test_validate_from_existing_credential_preserves_state(self):
        placeholder = ServiceCredential("aoc_x", state="placeholder")
        model = _CredentialModel.model_validate({"token": placeholder})
        assert model.token.is_placeholder
        # And literal state survives equally well.
        literal = ServiceCredential("ghp_y", state="literal")
        model = _CredentialModel.model_validate({"token": literal})
        assert model.token.is_literal

    def test_serialize_to_underlying_string(self):
        cred = ServiceCredential("aoc_xyz", state="placeholder")
        model = _CredentialModel(token=cred)
        # JSON serialization emits the bare string -- placeholder state
        # is in-process only.
        data = model.model_dump()
        assert data == {"token": "aoc_xyz"}

    def test_round_trip_through_json_loses_state(self):
        # Documented behavior: re-loading a serialized model gives a
        # ``literal`` credential, since JSON cannot carry the state
        # tag.  This is intentional -- placeholders never live on disk.
        cred = ServiceCredential("aoc_xyz", state="placeholder")
        model = _CredentialModel(token=cred)
        json_text = model.model_dump_json()
        reloaded = _CredentialModel.model_validate_json(json_text)
        assert reloaded.token.is_literal

    def test_validates_empty_string_as_literal_placeholder(self):
        # Empty strings round-trip through validation as the
        # structural-shim placeholder; see ``ServiceCredential.__new__``.
        model = _CredentialModel.model_validate({"token": ""})
        assert model.token == ""
        assert model.token.is_literal

    def test_repr_of_populated_model_redacts_value(self):
        # Pydantic models that hold a ``ServiceCredential`` should not
        # leak the credential value when rendered via ``repr``,
        # ``str``, or ``model_dump_json(... indent=2)``.  The first two
        # use our :meth:`ServiceCredential.__repr__`; ``model_dump_json``
        # uses the serialization path, but it always emits the bare
        # string -- so we don't claim it redacts (it cannot, because
        # JSON has to be re-loadable).  The point of this test is
        # logging / diagnostic surfaces, where ``repr`` is what shows.
        secret = "super-secret-value-should-never-appear-in-logs"
        model = _CredentialModel(token=ServiceCredential(secret))
        text = repr(model)
        assert secret not in text
        assert "literal" in text


# ---------------------------------------------------------------------------
# walk_credentials
# ---------------------------------------------------------------------------


class _NestedModel(BaseModel):
    name: str
    primary: ServiceCredential
    backup: ServiceCredential | None = None


class _OuterModel(BaseModel):
    nested: list[_NestedModel] = Field(default_factory=list)
    map: dict[str, ServiceCredential] = Field(default_factory=dict)


class TestWalkCredentials:
    def test_yields_top_level(self):
        cred = ServiceCredential("a")
        assert list(walk_credentials(cred)) == [cred]

    def test_yields_inside_pydantic_model(self):
        model = _NestedModel(
            name="github",
            primary=ServiceCredential("a"),
            backup=ServiceCredential("b", state="placeholder"),
        )
        result = list(walk_credentials(model))
        assert len(result) == 2
        assert {c.state for c in result} == {"literal", "placeholder"}

    def test_yields_through_lists_and_dicts(self):
        outer = _OuterModel(
            nested=[
                _NestedModel(name="x", primary=ServiceCredential("a")),
                _NestedModel(name="y", primary=ServiceCredential("b")),
            ],
            map={"k": ServiceCredential("c", state="placeholder")},
        )
        result = list(walk_credentials(outer))
        assert len(result) == 3

    def test_does_not_yield_plain_strings(self):
        # A plain ``str`` field in a model must not be reported as a
        # credential, even if its content looks credential-shaped.
        result = list(walk_credentials({"token": "ghp_abc"}))
        assert result == []

    def test_handles_cycles(self):
        # Pydantic models can't easily form cycles without help, but
        # plain dicts can.  Confirm no infinite loop.
        d: dict = {"cred": ServiceCredential("a")}
        d["self"] = d
        result = list(walk_credentials(d))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# assert_no_literal_credentials
# ---------------------------------------------------------------------------


class TestAuditAssertion:
    def test_passes_with_only_placeholders(self):
        model = _NestedModel(
            name="x",
            primary=ServiceCredential("p1", state="placeholder"),
            backup=ServiceCredential("p2", state="placeholder"),
        )
        # No exception.
        assert_no_literal_credentials(model)

    def test_passes_with_no_credentials(self):
        assert_no_literal_credentials({"foo": "bar"})

    def test_tolerates_empty_literal_credentials(self):
        # Empty literal credentials are structural shims (e.g., the
        # service-level forge config placeholder) and carry no real
        # auth material; the audit must not flag them.
        model = _NestedModel(
            name="x",
            primary=ServiceCredential(""),  # literal, but empty
            backup=ServiceCredential("real", state="placeholder"),
        )
        assert_no_literal_credentials(model)

    def test_fails_with_any_literal(self):
        model = _NestedModel(
            name="x",
            primary=ServiceCredential("real-token", state="literal"),
        )
        with pytest.raises(AssertionError, match="literal-state credential"):
            assert_no_literal_credentials(model)

    def test_failure_message_does_not_leak_value(self):
        model = _NestedModel(
            name="x",
            primary=ServiceCredential("super-secret-value", state="literal"),
        )
        try:
            assert_no_literal_credentials(model)
        except AssertionError as e:
            assert "super-secret-value" not in str(e)
            assert "literal len=" in str(e)
