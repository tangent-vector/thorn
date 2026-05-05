"""Unit tests for gateway git preflight helpers."""

from __future__ import annotations

from thorn.gateway._config import GatewayConfig
from thorn.gateway._preflight import (
    build_git_preflight_command,
    collect_event_source_preflight_problems,
    collect_forge_api_preflight_targets,
    collect_git_preflight_targets,
    git_preflight_failure_hint,
    redact_git_preflight_output,
)


def test_collect_git_preflight_targets_uses_resolved_clone_urls() -> None:
    config = GatewayConfig.model_validate({
        "projects": [
            {
                "name": "thorn",
                "url": "https://github.com/tangent-vector/thorn",
            },
            {
                "name": "multi",
                "forks": [
                    {
                        "name": "upstream",
                        "url": "https://gitlab.com/group/project",
                    },
                    {
                        "name": "mirror",
                        "url": "https://github.com/example/project",
                    },
                ],
            },
        ]
    })

    targets = collect_git_preflight_targets(
        config,
        project_filter="multi",
        fork_filter="mirror",
    )

    assert len(targets) == 1
    assert targets[0].project_name == "multi"
    assert targets[0].fork_name == "mirror"
    assert targets[0].clone_url == "https://github.com/example/project.git"


def test_event_source_preflight_requires_matching_forge_account() -> None:
    from thorn.core._agent import Agent
    from thorn.runtime import AgentID

    config = GatewayConfig.model_validate({
        "projects": [{
            "name": "thorn",
            "url": "https://gitlab.com/group/project",
        }]
    })
    agent = Agent(id=AgentID("coord"), name="coord")

    problems = collect_event_source_preflight_problems(config, [agent])

    assert len(problems) == 1
    assert problems[0].agent_id == AgentID("coord")
    assert problems[0].forge_name == "gitlab"
    assert "no account" in problems[0].reason


def test_event_source_preflight_reports_missing_credential_env(
    monkeypatch,
) -> None:
    from thorn.core._account import AgentAccountsConfig
    from thorn.core._agent import Agent
    from thorn.core._credentials import Credential
    from thorn.runtime import AgentID
    from thorn.tools.forge import GitLabAccountConfig

    monkeypatch.delenv("THORN_TEST_MISSING_GL_TOKEN", raising=False)
    config = GatewayConfig.model_validate({
        "projects": [{
            "name": "thorn",
            "url": "https://gitlab.com/group/project",
        }]
    })
    agent = Agent(
        id=AgentID("coord"),
        name="coord",
        accounts=AgentAccountsConfig(accounts=[
            GitLabAccountConfig(
                service="gitlab",
                credentials=[Credential(
                    kind="gitlab-pat",
                    env_var_name="THORN_TEST_MISSING_GL_TOKEN",
                )],
            ),
        ]),
    )

    problems = collect_event_source_preflight_problems(config, [agent])

    assert len(problems) == 1
    assert problems[0].forge_name == "gitlab"
    assert "THORN_TEST_MISSING_GL_TOKEN" in problems[0].reason


def test_collect_forge_api_preflight_targets_filters_project_fork() -> None:
    from thorn.core._account import AgentAccountsConfig
    from thorn.core._agent import Agent
    from thorn.core._credentials import Credential
    from thorn.runtime import AgentID
    from thorn.tools.forge import GitHubAccountConfig

    config = GatewayConfig.model_validate({
        "projects": [{
            "name": "multi",
            "forks": [
                {
                    "name": "upstream",
                    "url": "https://gitlab.com/group/project",
                },
                {
                    "name": "mirror",
                    "url": "https://github.com/example/project",
                },
            ],
        }]
    })
    agent = Agent(
        id=AgentID("coord"),
        name="coord",
        accounts=AgentAccountsConfig(accounts=[
            GitHubAccountConfig(
                service="github",
                credentials=[Credential(
                    kind="pat",
                    env_var_name="THORN_TEST_GH_TOKEN",
                )],
            ),
        ]),
    )

    targets = collect_forge_api_preflight_targets(
        config,
        [agent],
        project_filter="multi",
        fork_filter="mirror",
    )

    assert len(targets) == 1
    assert targets[0].agent_id == AgentID("coord")
    assert targets[0].forge_name == "github"
    assert targets[0].project_name == "multi"
    assert targets[0].fork_name == "mirror"
    assert targets[0].native_project_id == "example/project"


def test_read_only_command_uses_ls_remote_without_mutating_git() -> None:
    command = build_git_preflight_command(
        clone_url="https://gitlab.example.com/group/project.git",
        timeout_s=30,
    )

    assert "GIT_TERMINAL_PROMPT=0" in command
    assert "git ls-remote --exit-code" in command
    assert "git clone" not in command
    assert "git push" not in command


def test_write_check_command_pushes_and_deletes_temporary_branch() -> None:
    command = build_git_preflight_command(
        clone_url="https://gitlab.example.com/group/project.git",
        timeout_s=30,
        write_check_branch="thorn-preflight/abc123",
    )

    assert "commit --allow-empty" in command
    assert "HEAD:refs/heads/thorn-preflight/abc123" in command
    assert ":refs/heads/thorn-preflight/abc123" in command


def test_preflight_redaction_masks_tokens_headers_and_url_userinfo() -> None:
    output = "\n".join([
        "Authorization: Bearer glpat-secretvalue",
        "Proxy-Authorization: Basic aoc_secretvalue",
        "fatal: unable to access https://x:oc_proxysecret@gitlab.example/repo",
    ])

    redacted = redact_git_preflight_output(output)

    assert "glpat-secretvalue" not in redacted
    assert "aoc_secretvalue" not in redacted
    assert "oc_proxysecret" not in redacted
    assert "Authorization: <redacted>" in redacted
    assert "Proxy-Authorization: <redacted>" in redacted
    assert "https://<redacted>@gitlab.example/repo" in redacted


def test_preflight_failure_hints_cover_common_broker_failures() -> None:
    assert "Proxy" in (git_preflight_failure_hint("HTTP 407 from proxy") or "")
    assert "TLS" in (
        git_preflight_failure_hint("GnuTLS recv error: certificate failed")
        or ""
    )
    assert "DNS" in (
        git_preflight_failure_hint("Could not resolve host: gitlab.example")
        or ""
    )
