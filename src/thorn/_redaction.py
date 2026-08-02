"""Shared helpers for keeping diagnostic text free of credential material."""

from __future__ import annotations

import re

REDACTED_SECRET = "<redacted>"

_SECRET_FIELD_NAME_PATTERN = (
    r"[A-Za-z0-9_.-]*"
    r"(?:api[_-]?key|apikey|access[_-]?token|accesstoken|"
    r"token|secret|password|authorization)"
    r"[A-Za-z0-9_.-]*"
)

_HTTP_AUTH_HEADER_PATTERN = re.compile(
    r"(?im)\b(Proxy-Authorization|Authorization)\s*:\s*[^\r\n]*",
)
_QUOTED_SECRET_FIELD_PATTERN = re.compile(
    rf"(?i)([\"']{_SECRET_FIELD_NAME_PATTERN}[\"']\s*:\s*)"
    r"([\"'])"
    r"([^\"'\r\n]*)"
    r"([\"'])",
)
_KEY_VALUE_SECRET_FIELD_PATTERN = re.compile(
    rf"(?i)\b({_SECRET_FIELD_NAME_PATTERN})(\s*[:=]\s*)([^\s,;&}}]+)",
)
_HTTP_AUTH_SCHEME_PATTERN = re.compile(
    r"(?i)\b(Bearer|Basic)\s+([A-Za-z0-9._~+/=-]{8,})",
)
_URL_USERINFO_PATTERN = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://)([^/@\s]+)@",
)
_KNOWN_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"aoc_[A-Za-z0-9._~+/=-]{8,}"
    r"|oc_[A-Za-z0-9._~+/=-]{8,}"
    r"|glpat-[A-Za-z0-9._-]{8,}"
    r"|gh[opsru]_[A-Za-z0-9_]{8,}"
    r"|sk-[A-Za-z0-9._-]{8,}"
    r"|nvapi-[A-Za-z0-9._-]{8,}"
    r")(?![A-Za-z0-9])",
)


def redact_secrets(text: str) -> str:
    """Return *text* with credential-shaped substrings replaced.

    This intentionally targets the diagnostic surfaces Thorn owns:
    provider/broker error bodies, command traces, and object reprs.
    It is not a parser for any one upstream format; it is a final
    formatting guard that catches the token/header shapes we know can
    appear in provider and broker responses.
    """
    redacted = _HTTP_AUTH_HEADER_PATTERN.sub(
        lambda match: f"{match.group(1)}: {REDACTED_SECRET}",
        text,
    )
    redacted = _QUOTED_SECRET_FIELD_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}"
            f"{REDACTED_SECRET}{match.group(4)}"
        ),
        redacted,
    )
    redacted = _KEY_VALUE_SECRET_FIELD_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED_SECRET}",
        redacted,
    )
    redacted = _HTTP_AUTH_SCHEME_PATTERN.sub(
        lambda match: f"{match.group(1)} {REDACTED_SECRET}",
        redacted,
    )
    redacted = _URL_USERINFO_PATTERN.sub(
        lambda match: f"{match.group(1)}{REDACTED_SECRET}@",
        redacted,
    )
    return _KNOWN_TOKEN_PATTERN.sub(REDACTED_SECRET, redacted)


def redact_secret_snippet(text: str, *, max_chars: int) -> str:
    """Redact *text* first, then return a bounded diagnostic snippet."""
    redacted = redact_secrets(text)
    if len(redacted) <= max_chars:
        return redacted
    return f"{redacted[:max_chars]}..."
