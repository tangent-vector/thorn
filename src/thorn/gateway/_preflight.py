"""Preflight helpers for gateway sandbox/broker git connectivity."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from thorn.gateway._config import GatewayConfig, _resolve_forges_and_projects

_SECRET = "<redacted>"
_AUTH_HEADER_RE = re.compile(
    r"(?im)\b(?P<header>Authorization|Proxy-Authorization)\s*:\s*[^\r\n]+",
)
_URL_USERINFO_RE = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*://)"
    r"(?P<userinfo>[^/\s:@]+(?::[^/\s@]*)?@)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"\b(?:a?oc_[A-Za-z0-9._~+/=-]{3,}|glpat-[A-Za-z0-9._~+/=-]{6,}|"
    r"gh[pousr]_[A-Za-z0-9_]{6,})\b"
)


@dataclass(frozen=True)
class GitPreflightTarget:
    project_name: str
    fork_name: str
    clone_url: str


def collect_git_preflight_targets(
    config: GatewayConfig,
    *,
    project_filter: str | None = None,
    fork_filter: str | None = None,
) -> list[GitPreflightTarget]:
    """Return configured project clone URLs selected for git preflight."""
    _resolved_forges, resolved_projects = _resolve_forges_and_projects(config)
    targets: list[GitPreflightTarget] = []
    for project in resolved_projects:
        if project_filter is not None and project.name != project_filter:
            continue
        for fork in project.forks:
            if fork_filter is not None and fork.name != fork_filter:
                continue
            targets.append(
                GitPreflightTarget(
                    project_name=project.name,
                    fork_name=fork.name,
                    clone_url=fork.clone_url,
                )
            )
    return targets


def build_git_preflight_command(
    *,
    clone_url: str,
    timeout_s: int,
    write_check_branch: str | None = None,
) -> str:
    """Build the read-only or read-write git preflight shell command."""
    quoted_url = shlex.quote(clone_url)
    quoted_timeout = shlex.quote(str(timeout_s))
    lines = [
        "set -eu",
        'tmpdir="$(mktemp -d)"',
        'trap \'rm -rf "$tmpdir"\' EXIT',
        'cd "$tmpdir"',
        f"timeout {quoted_timeout} env GIT_TERMINAL_PROMPT=0 "
        f"git ls-remote --exit-code {quoted_url} HEAD",
    ]
    if write_check_branch is None:
        return "\n".join(lines)

    quoted_branch = shlex.quote(write_check_branch)
    lines.extend([
        "git init -q repo",
        "cd repo",
        f"git remote add origin {quoted_url}",
        "git -c user.name=thorn-preflight "
        "-c user.email=thorn-preflight@example.invalid "
        "commit --allow-empty -m 'thorn git preflight' >/dev/null",
        f"timeout {quoted_timeout} env GIT_TERMINAL_PROMPT=0 "
        f"git push origin HEAD:refs/heads/{quoted_branch}",
        f"timeout {quoted_timeout} env GIT_TERMINAL_PROMPT=0 "
        f"git push origin :refs/heads/{quoted_branch}",
    ])
    return "\n".join(lines)


def redact_git_preflight_output(output: str) -> str:
    redacted = _AUTH_HEADER_RE.sub(
        lambda match: f"{match.group('header')}: {_SECRET}",
        output,
    )
    redacted = _URL_USERINFO_RE.sub(
        lambda match: f"{match.group('scheme')}{_SECRET}@",
        redacted,
    )
    return _TOKEN_RE.sub(_SECRET, redacted)


def git_preflight_failure_hint(output: str) -> str | None:
    lowered = output.lower()
    if "proxy-authentication" in lowered or "proxy-authorization" in lowered:
        return (
            "Proxy authentication failed. Check that broker registration "
            "completed and that the sandbox is using Thorn's HTTPS proxy."
        )
    if "407" in lowered:
        return (
            "The broker proxy returned HTTP 407. Check broker credentials "
            "and Proxy-Authorization injection."
        )
    if any(
        marker in lowered
        for marker in (
            "certificate",
            "gnutls",
            "tls",
            "x509",
            "unknown authority",
            "self signed",
        )
    ):
        return (
            "TLS verification failed. Check the broker CA mount and "
            "ONECLI_HOST_CA_BUNDLE; use ONECLI_SKIP_VERIFY_HOSTS only "
            "for hosts where bypassing verification is acceptable."
        )
    if "could not resolve host" in lowered or "temporary failure in name" in lowered:
        return "DNS resolution failed from the sandbox/broker path."
    if "connection refused" in lowered or "no route to host" in lowered:
        return "Network egress failed from the sandbox/broker path."
    return None


__all__ = [
    "GitPreflightTarget",
    "build_git_preflight_command",
    "collect_git_preflight_targets",
    "git_preflight_failure_hint",
    "redact_git_preflight_output",
]
