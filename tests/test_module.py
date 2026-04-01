"""Tests for thorn._module — ModulePath type."""

from __future__ import annotations

import pytest

from thorn._module import ModulePath


class TestConstruction:
    def test_from_dotted_string(self):
        p = ModulePath("calc.parser")
        assert p.segments == ("calc", "parser")

    def test_from_single_segment(self):
        p = ModulePath("calc")
        assert p.segments == ("calc",)

    def test_from_tuple(self):
        p = ModulePath(("calc", "parser"))
        assert p.segments == ("calc", "parser")

    def test_from_empty_tuple(self):
        p = ModulePath(())
        assert p.segments == ()
        assert p.is_root

    def test_from_empty_string(self):
        p = ModulePath("")
        assert p.is_root

    def test_from_module_path_identity(self):
        original = ModulePath("calc.parser")
        copy = ModulePath(original)
        assert copy == original
        assert copy.segments is original.segments

    def test_from_underscore_is_root(self):
        p = ModulePath("_")
        assert p.is_root

    def test_leading_underscore_stripped(self):
        p = ModulePath("_.calc.parser")
        assert p.segments == ("calc", "parser")

    def test_underscore_dot_is_root(self):
        p = ModulePath("_.")
        assert p.is_root

    def test_default_arg_is_root(self):
        p = ModulePath()
        assert p.is_root

    def test_root_classmethod(self):
        p = ModulePath.root()
        assert p.is_root
        assert p.segments == ()


class TestValidation:
    def test_double_dot_raises(self):
        with pytest.raises(ValueError, match="empty module-path segment"):
            ModulePath("a..b")

    def test_trailing_dot_raises(self):
        with pytest.raises(ValueError, match="empty module-path segment"):
            ModulePath("a.b.")

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError, match="invalid module-path segment"):
            ModulePath("a.b-c")

    def test_starts_with_digit_raises(self):
        with pytest.raises(ValueError, match="invalid module-path segment"):
            ModulePath("123")

    def test_underscore_segment_reserved(self):
        with pytest.raises(ValueError, match="reserved for the root"):
            ModulePath("calc._")

    def test_underscore_in_middle_reserved(self):
        with pytest.raises(ValueError, match="reserved for the root"):
            ModulePath("calc._.parser")

    def test_tuple_with_invalid_segment_raises(self):
        with pytest.raises(ValueError, match="invalid module-path segment"):
            ModulePath(("calc", "bad-name"))

    def test_tuple_with_non_string_raises(self):
        with pytest.raises(TypeError, match="must be strings"):
            ModulePath((42,))  # type: ignore[arg-type]

    def test_wrong_source_type_raises(self):
        with pytest.raises(TypeError, match="requires str, tuple, or ModulePath"):
            ModulePath(42)  # type: ignore[arg-type]


class TestDisplay:
    def test_str_dotted(self):
        assert str(ModulePath("calc.parser")) == "calc.parser"

    def test_str_single(self):
        assert str(ModulePath("calc")) == "calc"

    def test_str_root(self):
        assert str(ModulePath.root()) == "_"

    def test_repr(self):
        assert repr(ModulePath("calc.parser")) == "ModulePath('calc.parser')"

    def test_repr_root(self):
        assert repr(ModulePath.root()) == "ModulePath('_')"

    def test_format(self):
        p = ModulePath("calc.parser")
        assert f"{p}" == "calc.parser"
        assert f"module={p}" == "module=calc.parser"

    def test_format_in_template(self):
        p = ModulePath("calc")
        result = "Working on {module}.".format_map({"module": p})
        assert result == "Working on calc."


class TestProperties:
    def test_child(self):
        p = ModulePath("calc")
        c = p.child("parser")
        assert c.segments == ("calc", "parser")

    def test_child_from_root(self):
        c = ModulePath.root().child("calc")
        assert c.segments == ("calc",)

    def test_child_validates(self):
        with pytest.raises(ValueError):
            ModulePath("calc").child("bad-name")

    def test_parent(self):
        p = ModulePath("calc.parser.lexer")
        assert p.parent == ModulePath("calc.parser")

    def test_parent_of_single_is_root(self):
        p = ModulePath("calc")
        assert p.parent == ModulePath.root()

    def test_parent_of_root_is_none(self):
        assert ModulePath.root().parent is None

    def test_name(self):
        assert ModulePath("calc.parser").name == "parser"

    def test_name_single(self):
        assert ModulePath("calc").name == "calc"

    def test_name_root(self):
        assert ModulePath.root().name == ""

    def test_is_root(self):
        assert ModulePath.root().is_root
        assert not ModulePath("calc").is_root

    def test_depth(self):
        assert ModulePath.root().depth == 0
        assert ModulePath("calc").depth == 1
        assert ModulePath("calc.parser").depth == 2


class TestIteration:
    def test_iter_segments(self):
        assert list(ModulePath("calc.parser")) == ["calc", "parser"]

    def test_iter_root_empty(self):
        assert list(ModulePath.root()) == []


class TestEquality:
    def test_equal_paths(self):
        assert ModulePath("calc.parser") == ModulePath(("calc", "parser"))

    def test_not_equal(self):
        assert ModulePath("calc") != ModulePath("parser")

    def test_not_equal_to_string(self):
        assert ModulePath("calc") != "calc"

    def test_hash_consistent(self):
        a = ModulePath("calc.parser")
        b = ModulePath("calc.parser")
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self):
        d = {ModulePath("calc"): 1, ModulePath("parser"): 2}
        assert d[ModulePath("calc")] == 1

    def test_usable_in_set(self):
        s = {ModulePath("calc"), ModulePath("calc"), ModulePath("parser")}
        assert len(s) == 2

    def test_frozen(self):
        p = ModulePath("calc")
        with pytest.raises(AttributeError):
            p.segments = ("other",)  # type: ignore[misc]


class TestRoundTrip:
    """Constructing from str(path) should yield an equal path."""

    def test_roundtrip_dotted(self):
        p = ModulePath("calc.parser.lexer")
        assert ModulePath(str(p)) == p

    def test_roundtrip_root(self):
        p = ModulePath.root()
        assert ModulePath(str(p)) == p

    def test_roundtrip_single(self):
        p = ModulePath("calc")
        assert ModulePath(str(p)) == p

    def test_leading_underscore_equivalent(self):
        assert ModulePath("_.calc.parser") == ModulePath("calc.parser")
