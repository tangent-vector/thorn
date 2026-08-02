"""Preflight helpers for gateway readiness checks."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from dataclasses import dataclass
from typing import Any

import httpx

from thorn.core._account import AccountConfig, find_credential
from thorn.core._credentials import CredentialMissingError
from thorn.core._messages import UserMessage
from thorn.core._provider import LLMProvider, OpenAIProvider
from thorn.core.errors import ProviderError
from thorn.gateway._config import (
    ForgeSpec,
    GatewayConfig,
    _resolve_forges_and_projects,
)
from thorn.runtime import AgentID

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
_LLM_PREFLIGHT_MAX_COMPLETION_TOKENS = 64


@dataclass(frozen=True)
class GitPreflightTarget:
    project_name: str
    fork_name: str
    clone_url: str


@dataclass(frozen=True)
class EventSourcePreflightProblem:
    agent_id: AgentID
    forge_name: str
    reason: str


@dataclass(frozen=True)
class ForgeAPIPreflightTarget:
    agent_id: AgentID
    account: AccountConfig
    forge_name: str
    project_name: str
    fork_name: str
    native_project_id: str


@dataclass(frozen=True)
class LLMPreflightTarget:
    agent_id: AgentID
    provider: LLMProvider


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


def collect_event_source_preflight_problems(
    config: GatewayConfig,
    agents: list[Any],
    *,
    project_filter: str | None = None,
    fork_filter: str | None = None,
) -> list[EventSourcePreflightProblem]:
    """Return static problems that would prevent source inference.

    The gateway's source inference path intentionally does not poll
    upstream services.  These checks keep ``thorn serve preflight`` just
    as non-destructive while still catching the common first-run
    failures: no matching forge account, wrong credential kind, or a
    credential env var missing from the gateway process.
    """
    selected_forges = _selected_forge_specs_by_name(
        config,
        project_filter=project_filter,
        fork_filter=fork_filter,
    )
    if not selected_forges:
        return []

    problems: list[EventSourcePreflightProblem] = []
    for agent in agents:
        agent_id = _agent_id_for_preflight(agent)
        matching_accounts = _matching_forge_accounts(agent, selected_forges)
        if not matching_accounts:
            forge_names = ", ".join(sorted(selected_forges))
            problems.append(EventSourcePreflightProblem(
                agent_id=agent_id,
                forge_name=forge_names,
                reason=(
                    "agent has no account for the selected forge services "
                    f"({forge_names})"
                ),
            ))
            continue

        for account, forge_spec in matching_accounts:
            credential_kind = _event_source_credential_kind(forge_spec.type)
            if credential_kind is None:
                problems.append(EventSourcePreflightProblem(
                    agent_id=agent_id,
                    forge_name=forge_spec.name,
                    reason=(
                        "no event-source inference strategy exists for "
                        f"forge type {forge_spec.type!r}"
                    ),
                ))
                continue

            credential = find_credential(account, kind=credential_kind)
            if credential is None:
                available = sorted({c.kind for c in account.credentials})
                problems.append(EventSourcePreflightProblem(
                    agent_id=agent_id,
                    forge_name=forge_spec.name,
                    reason=(
                        f"account has no {credential_kind!r} credential "
                        "needed for event-source polling; available kinds: "
                        f"{', '.join(available) if available else '(none)'}"
                    ),
                ))
                continue

            try:
                credential.read_value()
            except CredentialMissingError as exc:
                problems.append(EventSourcePreflightProblem(
                    agent_id=agent_id,
                    forge_name=forge_spec.name,
                    reason=str(exc),
                ))

    return problems


def collect_forge_api_preflight_targets(
    config: GatewayConfig,
    agents: list[Any],
    *,
    project_filter: str | None = None,
    fork_filter: str | None = None,
) -> list[ForgeAPIPreflightTarget]:
    """Return direct forge API probes selected by the preflight filters."""
    forge_specs_by_name = _selected_forge_specs_by_name(
        config,
        project_filter=project_filter,
        fork_filter=fork_filter,
    )
    if not forge_specs_by_name:
        return []

    _resolved_forges, resolved_projects = _resolve_forges_and_projects(config)
    targets: list[ForgeAPIPreflightTarget] = []
    seen: set[tuple[AgentID, str, str]] = set()
    for project in resolved_projects:
        if project_filter is not None and project.name != project_filter:
            continue
        for fork in project.forks:
            if fork_filter is not None and fork.name != fork_filter:
                continue
            if fork.forge_name not in forge_specs_by_name:
                continue
            for agent in agents:
                agent_id = _agent_id_for_preflight(agent)
                for account, _forge_spec in _matching_forge_accounts(
                    agent,
                    forge_specs_by_name,
                ):
                    if account.service != fork.forge_name:
                        continue
                    key = (agent_id, fork.forge_name, fork.native_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    targets.append(ForgeAPIPreflightTarget(
                        agent_id=agent_id,
                        account=account,
                        forge_name=fork.forge_name,
                        project_name=project.name,
                        fork_name=fork.name,
                        native_project_id=fork.native_id,
                    ))
    return targets


async def probe_llm_preflight_target(
    target: LLMPreflightTarget,
    *,
    timeout_s: int,
) -> None:
    """Verify that an agent's effective provider can stream a tiny response."""
    if timeout_s < 1:
        raise ValueError("LLM preflight timeout must be at least one second")

    if isinstance(target.provider, OpenAIProvider):
        await _probe_openai_llm_preflight_target(
            target.provider,
            timeout_s=timeout_s,
        )
        return

    try:
        async with asyncio.timeout(float(timeout_s)):
            async for _chunk in target.provider.complete(
                [
                    (
                        "Thorn preflight is checking that the configured "
                        "LLM model can stream a minimal chat completion. "
                        "Reply with OK."
                    )
                ],
                [],
                [UserMessage(content="Reply with OK.")],
            ):
                pass
    except TimeoutError as exc:
        raise ProviderError(
            f"LLM preflight probe timed out after {timeout_s}s"
        ) from exc


async def _probe_openai_llm_preflight_target(
    provider: OpenAIProvider,
    *,
    timeout_s: int,
) -> None:
    """Issue a cheap OpenAI-compatible streaming readiness request.

    The runtime provider path asks for usage in the final SSE trailer and
    inherits operator completion budgets.  Both are useful for real agent turns
    but can make a one-token preflight wait on provider-side accounting or
    reasoning latency.  This probe validates the configured endpoint, key, model,
    and streaming transport, then returns as soon as a real choice chunk arrives.
    """
    body: dict[str, Any] = {
        "model": provider.config.model_name,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "stream": True,
        "max_completion_tokens": _LLM_PREFLIGHT_MAX_COMPLETION_TOKENS,
    }
    try:
        async with asyncio.timeout(float(timeout_s)):
            async with provider._client.stream(  # noqa: SLF001
                "POST",
                "/chat/completions",
                json=body,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise ProviderError(
                        f"provider returned HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("choices"):
                        return
    except TimeoutError as exc:
        raise ProviderError(
            f"LLM preflight probe timed out after {timeout_s}s"
        ) from exc
    except httpx.TransportError as exc:
        raise ProviderError(f"transport error talking to provider: {exc}") from exc

    raise ProviderError("LLM preflight stream ended before a response chunk arrived")


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


def _agent_id_for_preflight(agent: Any) -> AgentID:
    agent_id = getattr(agent, "id", None)
    if isinstance(agent_id, AgentID):
        return agent_id
    if agent_id is not None:
        return AgentID(str(agent_id))
    name = getattr(agent, "name", None) or "unknown"
    return AgentID(str(name))


def _selected_forge_specs_by_name(
    config: GatewayConfig,
    *,
    project_filter: str | None,
    fork_filter: str | None,
) -> dict[str, ForgeSpec]:
    forge_specs, resolved_projects = _resolve_forges_and_projects(config)
    selected_forge_names: set[str] = set()
    for project in resolved_projects:
        if project_filter is not None and project.name != project_filter:
            continue
        for fork in project.forks:
            if fork_filter is not None and fork.name != fork_filter:
                continue
            selected_forge_names.add(fork.forge_name)
    return {
        forge_spec.name: forge_spec
        for forge_spec in forge_specs
        if forge_spec.name in selected_forge_names
    }


def _matching_forge_accounts(
    agent: Any,
    forge_specs_by_name: dict[str, ForgeSpec],
) -> list[tuple[AccountConfig, ForgeSpec]]:
    accounts = getattr(agent, "accounts", None)
    if accounts is None:
        return []

    matches: list[tuple[AccountConfig, ForgeSpec]] = []
    for account in accounts.accounts:
        forge_spec = forge_specs_by_name.get(account.service)
        if forge_spec is None:
            continue
        matches.append((account, forge_spec))
    return matches


def _event_source_credential_kind(forge_type: str) -> str | None:
    if forge_type == "github":
        return "pat"
    if forge_type == "gitlab":
        return "gitlab-pat"
    return None


__all__ = [
    "EventSourcePreflightProblem",
    "ForgeAPIPreflightTarget",
    "GitPreflightTarget",
    "build_git_preflight_command",
    "collect_event_source_preflight_problems",
    "collect_forge_api_preflight_targets",
    "collect_git_preflight_targets",
    "git_preflight_failure_hint",
    "redact_git_preflight_output",
]
