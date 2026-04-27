"""Tests for `thorn.runtime._skill_md`.

Covers the documented frontmatter contract case-by-case so that any
future change to either the spec (e.g. new required fields) or the
underlying YAML library is forced to update the relevant case
explicitly rather than silently shifting behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thorn.runtime._skill_md import (
    MalformedFrontmatterError,
    MissingDescriptionError,
    MissingFrontmatterError,
    ParsedSkillMd,
    SkillMdError,
    parse_skill_md,
)


_DUMMY_PATH = Path("/tmp/SKILL.md")


# ---------------------------------------------------------------------------
# Well-formed input
# ---------------------------------------------------------------------------

class TestWellFormed:
    def test_minimal_description_only(self) -> None:
        text = (
            "---\n"
            "description: Keep PRs merge-ready.\n"
            "---\n"
            "Body line one.\n"
        )
        result = parse_skill_md(_DUMMY_PATH, text)
        assert result == ParsedSkillMd(
            description="Keep PRs merge-ready.",
            body="Body line one.\n",
        )

    def test_extra_keys_are_ignored(self) -> None:
        # Forward-compat: unknown keys must not break parsing.
        text = (
            "---\n"
            "description: One-liner.\n"
            "allowed-tools: [read_file, Shell]\n"
            "model: claude-opus\n"
            "---\n"
            "# Heading\n\nBody.\n"
        )
        result = parse_skill_md(_DUMMY_PATH, text)
        assert result.description == "One-liner."
        assert result.body == "# Heading\n\nBody.\n"

    def test_description_is_whitespace_stripped(self) -> None:
        text = (
            "---\n"
            "description: '   surrounding spaces   '\n"
            "---\n"
        )
        result = parse_skill_md(_DUMMY_PATH, text)
        assert result.description == "surrounding spaces"

    def test_empty_body_is_supported(self) -> None:
        text = (
            "---\n"
            "description: Just metadata.\n"
            "---\n"
        )
        result = parse_skill_md(_DUMMY_PATH, text)
        assert result.description == "Just metadata."
        assert result.body == ""

    def test_crlf_line_endings_are_tolerated(self) -> None:
        # Files written on Windows (or copy-pasted from the web) often
        # use CRLF; the fence detector strips the trailing CR before
        # comparing so they parse identically.
        text = (
            "---\r\n"
            "description: CRLF works.\r\n"
            "---\r\n"
            "Body.\r\n"
        )
        result = parse_skill_md(_DUMMY_PATH, text)
        assert result.description == "CRLF works."

    def test_multi_line_body_after_fence(self) -> None:
        text = (
            "---\n"
            "description: Has body.\n"
            "---\n"
            "Line 1.\n"
            "Line 2.\n"
            "Line 3.\n"
        )
        result = parse_skill_md(_DUMMY_PATH, text)
        assert result.body == "Line 1.\nLine 2.\nLine 3.\n"


# ---------------------------------------------------------------------------
# Missing frontmatter
# ---------------------------------------------------------------------------

class TestMissingFrontmatter:
    def test_empty_file_raises_missing(self) -> None:
        with pytest.raises(MissingFrontmatterError):
            parse_skill_md(_DUMMY_PATH, "")

    def test_no_opening_fence_raises_missing(self) -> None:
        text = "# A skill\n\ndescription: Not in frontmatter.\n"
        with pytest.raises(MissingFrontmatterError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_blank_line_before_fence_raises_missing(self) -> None:
        # Per spec the fence must be on the very first line.
        text = "\n---\ndescription: x\n---\n"
        with pytest.raises(MissingFrontmatterError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_almost_fence_does_not_count(self) -> None:
        # ``----`` (four dashes) is not a fence; common typo.
        text = "----\ndescription: x\n----\n"
        with pytest.raises(MissingFrontmatterError):
            parse_skill_md(_DUMMY_PATH, text)


# ---------------------------------------------------------------------------
# Malformed frontmatter
# ---------------------------------------------------------------------------

class TestMalformedFrontmatter:
    def test_unclosed_fence_raises_malformed(self) -> None:
        text = "---\ndescription: Never closed.\nbody continues forever\n"
        with pytest.raises(MalformedFrontmatterError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_invalid_yaml_raises_malformed(self) -> None:
        text = (
            "---\n"
            "description: 'unterminated string\n"
            "---\n"
        )
        with pytest.raises(MalformedFrontmatterError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_yaml_scalar_raises_malformed(self) -> None:
        # Frontmatter must be a mapping; a bare scalar is rejected.
        text = "---\njust a string\n---\n"
        with pytest.raises(MalformedFrontmatterError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_yaml_list_raises_malformed(self) -> None:
        text = "---\n- one\n- two\n---\n"
        with pytest.raises(MalformedFrontmatterError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_empty_frontmatter_raises_malformed(self) -> None:
        text = "---\n---\n"
        with pytest.raises(MalformedFrontmatterError):
            parse_skill_md(_DUMMY_PATH, text)


# ---------------------------------------------------------------------------
# Missing/unusable description
# ---------------------------------------------------------------------------

class TestMissingDescription:
    def test_no_description_key_raises(self) -> None:
        text = "---\nname: foo\n---\nbody\n"
        with pytest.raises(MissingDescriptionError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_non_string_description_raises(self) -> None:
        text = "---\ndescription: 42\n---\n"
        with pytest.raises(MissingDescriptionError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_null_description_raises(self) -> None:
        text = "---\ndescription:\n---\n"
        with pytest.raises(MissingDescriptionError):
            parse_skill_md(_DUMMY_PATH, text)

    def test_whitespace_only_description_raises(self) -> None:
        text = "---\ndescription: '   '\n---\n"
        with pytest.raises(MissingDescriptionError):
            parse_skill_md(_DUMMY_PATH, text)


# ---------------------------------------------------------------------------
# Error type hierarchy (so callers can do `except SkillMdError`)
# ---------------------------------------------------------------------------

class TestErrorHierarchy:
    @pytest.mark.parametrize(
        "subclass",
        [
            MissingFrontmatterError,
            MalformedFrontmatterError,
            MissingDescriptionError,
        ],
    )
    def test_all_subclasses_are_skill_md_errors(
        self, subclass: type,
    ) -> None:
        assert issubclass(subclass, SkillMdError)

    def test_errors_carry_path(self) -> None:
        path = Path("/skills/foo/SKILL.md")
        try:
            parse_skill_md(path, "no frontmatter at all")
        except SkillMdError as exc:
            assert exc.path == path
            assert str(path) in str(exc)
        else:
            pytest.fail("expected SkillMdError")
