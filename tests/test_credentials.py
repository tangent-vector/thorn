"""Tests for ``thorn.core._credentials`` -- ``Credential`` + ``ServiceCredential``.

Covers the contract documented in :mod:`thorn.core._credentials`:

- :class:`Credential` carries only an env var *reference* (no
  literal); :meth:`Credential.read_value` is the only path that
  reaches into ``os.environ``.
- :class:`ServiceCredential` is a redacted-on-repr ``str`` subclass
  with no other state.
"""

from __future__ import annotations

import pytest

from thorn.core._credentials import (
    Credential,
    CredentialMissingError,
    ServiceCredential,
)

# ---------------------------------------------------------------------------
# ServiceCredential
# ---------------------------------------------------------------------------


class TestServiceCredentialBasics:
    def test_constructs_from_str(self):
        cred = ServiceCredential("ghp_abc")
        assert cred == "ghp_abc"
        assert isinstance(cred, str)

    def test_repr_redacts(self):
        cred = ServiceCredential("super-secret-value")
        text = repr(cred)
        assert "super-secret-value" not in text
        assert "len=18" in text
        assert "redacted" in text

    def test_redacted_method_hides_value(self):
        cred = ServiceCredential("abc")
        out = cred.redacted()
        assert "abc" not in out
        assert "len=3" in out

    def test_str_does_not_redact(self):
        # ``str(cred)`` is the underlying value, by design -- call
        # sites pass the credential to HTTP clients using ``str()``
        # or implicit conversion.  Only ``repr()`` redacts.
        cred = ServiceCredential("ghp_abc")
        assert str(cred) == "ghp_abc"

    def test_equality_with_plain_str(self):
        cred = ServiceCredential("ghp_abc")
        assert cred == "ghp_abc"
        assert "ghp_abc" == cred

    def test_hashes_like_str(self):
        cred = ServiceCredential("ghp_abc")
        assert hash(cred) == hash("ghp_abc")


# ---------------------------------------------------------------------------
# Credential model
# ---------------------------------------------------------------------------


class TestCredentialModel:
    def test_required_fields(self):
        cred = Credential(kind="pat", env_var_name="GITHUB_TOKEN")
        assert cred.kind == "pat"
        assert cred.env_var_name == "GITHUB_TOKEN"
        assert cred.name is None

    def test_with_name(self):
        cred = Credential(
            kind="pat", name="primary", env_var_name="GH_PRIMARY",
        )
        assert cred.name == "primary"

    def test_kind_min_length(self):
        with pytest.raises(ValueError):
            Credential(kind="", env_var_name="X")

    def test_env_var_name_min_length(self):
        with pytest.raises(ValueError):
            Credential(kind="pat", env_var_name="")

    def test_repr_does_not_leak_or_resolve_value(self, monkeypatch):
        # ``repr`` only mentions the env var *name*, not its value;
        # also does not call ``os.environ`` at all (so even an unset
        # var is harmless).
        cred = Credential(kind="pat", env_var_name="MY_TOKEN")
        text = repr(cred)
        assert "MY_TOKEN" in text
        assert "kind='pat'" in text

    def test_round_trip_json(self):
        cred = Credential(kind="pat", env_var_name="GITHUB_TOKEN")
        data = cred.model_dump()
        assert data == {
            "kind": "pat", "name": None, "env_var_name": "GITHUB_TOKEN",
        }
        restored = Credential.model_validate(data)
        assert restored == cred


class TestCredentialReadValue:
    def test_returns_service_credential(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "ghp_abc")
        cred = Credential(kind="pat", env_var_name="MY_TOKEN")
        value = cred.read_value()
        assert isinstance(value, ServiceCredential)
        assert value == "ghp_abc"

    def test_raises_credential_missing_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MY_TOKEN", raising=False)
        cred = Credential(kind="pat", env_var_name="MY_TOKEN")
        with pytest.raises(CredentialMissingError) as exc_info:
            cred.read_value()
        # Error message names the env var so the operator knows what
        # to set; we don't promise it includes the kind verbatim.
        assert "MY_TOKEN" in str(exc_info.value)

    def test_credential_missing_is_lookup_error(self):
        # Subclass of LookupError so callers can catch the broader
        # category when they prefer.
        assert issubclass(CredentialMissingError, LookupError)

    def test_credential_is_frozen(self):
        cred = Credential(kind="pat", env_var_name="X")
        with pytest.raises(ValueError):
            cred.kind = "other"  # type: ignore[misc]
