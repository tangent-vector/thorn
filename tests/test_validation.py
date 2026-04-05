"""Tests for thorn.core._validation — ValidationRule type."""

from __future__ import annotations

import pytest

from thorn.core._validation import ValidationRule


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the global registry around each test."""
    saved = dict(ValidationRule._registry)
    yield
    ValidationRule._registry.clear()
    ValidationRule._registry.update(saved)


class TestCreation:
    def test_basic(self):
        def check():
            pass

        rule = ValidationRule("my_rule", check=check)
        assert rule.name == "my_rule"
        assert rule.check is check

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ValidationRule("", check=lambda: None)


class TestRegistry:
    def test_auto_registered(self):
        def check():
            pass

        rule = ValidationRule("reg_test", check=check)
        assert ValidationRule.lookup("reg_test") is rule

    def test_lookup_unknown_returns_none(self):
        assert ValidationRule.lookup("nonexistent") is None

    def test_re_registration_overwrites(self):
        def check_v1():
            pass

        def check_v2():
            pass

        rule1 = ValidationRule("overwrite_test", check=check_v1)
        rule2 = ValidationRule("overwrite_test", check=check_v2)
        assert ValidationRule.lookup("overwrite_test") is rule2
        assert ValidationRule.lookup("overwrite_test").check is check_v2


class TestHashAndEquality:
    def test_same_name_equal(self):
        def check():
            pass

        a = ValidationRule("eq_test_a", check=check)
        b = ValidationRule("eq_test_b", check=check)
        # Re-create with same name — needs overwrite since auto-registered
        c = ValidationRule("eq_test_a", check=lambda: None)
        assert a == c
        assert a != b

    def test_not_equal_to_string(self):
        rule = ValidationRule("ne_test", check=lambda: None)
        assert rule != "ne_test"

    def test_hash_by_name(self):
        rule = ValidationRule("hash_test", check=lambda: None)
        assert hash(rule) == hash("hash_test")

    def test_usable_in_set(self):
        a = ValidationRule("set_a", check=lambda: None)
        b = ValidationRule("set_b", check=lambda: None)
        s = frozenset({a, b})
        assert len(s) == 2
        assert a in s

    def test_usable_as_dict_key(self):
        rule = ValidationRule("dict_test", check=lambda: None)
        d = {rule: 42}
        assert d[rule] == 42


class TestRepr:
    def test_repr(self):
        rule = ValidationRule("repr_test", check=lambda: None)
        assert repr(rule) == "ValidationRule('repr_test')"
