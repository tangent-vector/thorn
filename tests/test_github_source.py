"""Tests for GitHubNotificationsSource (repository events API)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from thorn.gateway._event import IncomingEvent
from thorn.runtime import SessionKey
from thorn.tools._github_connection import GitHubPatAuth


# ---------------------------------------------------------------------------
# Helpers under test (private but stable for unit tests)
# ---------------------------------------------------------------------------


class TestSessionKeyForEvent:
    def test_stable_key(self) -> None:
        from thorn.gateway._routing import route_github_event

        k = route_github_event(
            repo_id=42, event_type="IssuesEvent", event_id="evt-123",
        )
        assert k == SessionKey("github/42/issuesevent/evt-123")


class TestPayloadSummary:
    def test_issues_event(self) -> None:
        from thorn.gateway.sources._github import _payload_summary

        payload = {
            "action": "opened",
            "issue": {"number": 3, "title": "Bug"},
        }
        s = _payload_summary("IssuesEvent", payload)
        assert "3" in s and "Bug" in s

    def test_empty_payload(self) -> None:
        from thorn.gateway.sources._github import _payload_summary

        assert "no payload" in _payload_summary("X", {}).lower()


class TestExtractNoteable:
    """Test _extract_noteable for each GitHub event type."""

    def test_issues_event(self) -> None:
        from thorn.gateway._routing import NoteableKind
        from thorn.gateway.sources._github import _extract_noteable

        n = _extract_noteable("IssuesEvent", {
            "action": "opened",
            "issue": {"number": 7, "title": "Bug"},
        })
        assert n is not None
        assert n.kind is NoteableKind.ISSUE
        assert n.number == 7

    def test_issue_comment_on_issue(self) -> None:
        from thorn.gateway._routing import NoteableKind
        from thorn.gateway.sources._github import _extract_noteable

        n = _extract_noteable("IssueCommentEvent", {
            "comment": {"body": "hello"},
            "issue": {"number": 3},
        })
        assert n is not None
        assert n.kind is NoteableKind.ISSUE
        assert n.number == 3

    def test_issue_comment_on_pull_request(self) -> None:
        from thorn.gateway._routing import NoteableKind
        from thorn.gateway.sources._github import _extract_noteable

        n = _extract_noteable("IssueCommentEvent", {
            "comment": {"body": "lgtm"},
            "issue": {
                "number": 5,
                "pull_request": {"url": "https://..."},
            },
        })
        assert n is not None
        assert n.kind is NoteableKind.CHANGE_REQUEST
        assert n.number == 5

    def test_pull_request_event(self) -> None:
        from thorn.gateway._routing import NoteableKind
        from thorn.gateway.sources._github import _extract_noteable

        n = _extract_noteable("PullRequestEvent", {
            "action": "opened",
            "pull_request": {"number": 10, "title": "Fix"},
        })
        assert n is not None
        assert n.kind is NoteableKind.CHANGE_REQUEST
        assert n.number == 10

    def test_pull_request_review_event(self) -> None:
        from thorn.gateway._routing import NoteableKind
        from thorn.gateway.sources._github import _extract_noteable

        n = _extract_noteable("PullRequestReviewEvent", {
            "review": {"state": "approved"},
            "pull_request": {"number": 10},
        })
        assert n is not None
        assert n.kind is NoteableKind.CHANGE_REQUEST
        assert n.number == 10

    def test_pull_request_review_comment_event(self) -> None:
        from thorn.gateway._routing import NoteableKind
        from thorn.gateway.sources._github import _extract_noteable

        n = _extract_noteable("PullRequestReviewCommentEvent", {
            "comment": {"body": "nit"},
            "pull_request": {"number": 10},
        })
        assert n is not None
        assert n.kind is NoteableKind.CHANGE_REQUEST
        assert n.number == 10

    def test_push_event_returns_none(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable

        assert _extract_noteable("PushEvent", {"ref": "refs/heads/main"}) is None

    def test_create_event_returns_none(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable

        assert _extract_noteable("CreateEvent", {"ref_type": "branch"}) is None

    def test_delete_event_returns_none(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable

        assert _extract_noteable("DeleteEvent", {"ref_type": "branch"}) is None

    def test_malformed_payload_returns_none(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable

        assert _extract_noteable("IssuesEvent", {}) is None
        assert _extract_noteable("IssuesEvent", {"issue": {}}) is None
        assert _extract_noteable("PullRequestEvent", {}) is None


class TestMakeIncomingEvent:
    def test_metadata(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        repo = MagicMock()
        repo.id = 99
        repo.full_name = "o/r"
        repo.clone_url = "https://github.com/o/r.git"
        repo.default_branch = "main"
        repo.html_url = "https://github.com/o/r"

        ev = _make_incoming_event(
            repo=repo,
            event_type="IssuesEvent",
            event_id="e1",
            actor_login="alice",
            created_at="2020-01-01T00:00:00Z",
            payload={"action": "opened", "issue": {"number": 1, "title": "Hi"}},
        )
        assert ev.source == "github"
        assert ev.metadata["event_id"] == "e1"
        assert ev.metadata["event_type"] == "IssuesEvent"
        assert ev.metadata["repo_id"] == 99
        assert ev.metadata["actor_login"] == "alice"
        assert "GitHub repository activity" in ev.content

    def test_issues_event_routes_to_issue_session(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        repo = MagicMock()
        repo.id = 42
        repo.full_name = "o/r"
        repo.clone_url = "https://github.com/o/r.git"
        repo.default_branch = "main"
        repo.html_url = "https://github.com/o/r"

        ev = _make_incoming_event(
            repo=repo,
            event_type="IssuesEvent",
            event_id="e1",
            actor_login="alice",
            created_at="2020-01-01T00:00:00Z",
            payload={"action": "opened", "issue": {"number": 5, "title": "Bug"}},
        )
        assert ev.session_key == SessionKey("github/42/issue/5")

    def test_pull_request_event_routes_to_change_request_session(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        repo = MagicMock()
        repo.id = 42
        repo.full_name = "o/r"
        repo.clone_url = "https://github.com/o/r.git"
        repo.default_branch = "main"
        repo.html_url = "https://github.com/o/r"

        ev = _make_incoming_event(
            repo=repo,
            event_type="PullRequestEvent",
            event_id="e2",
            actor_login="alice",
            created_at="2020-01-01T00:00:00Z",
            payload={"action": "opened", "pull_request": {"number": 3, "title": "Fix"}},
        )
        assert ev.session_key == SessionKey("github/42/change-request/3")

    def test_issue_and_comment_share_session(self) -> None:
        """IssuesEvent and IssueCommentEvent for the same issue produce the same key."""
        from thorn.gateway.sources._github import _make_incoming_event

        repo = MagicMock()
        repo.id = 42
        repo.full_name = "o/r"
        repo.clone_url = "https://github.com/o/r.git"
        repo.default_branch = "main"
        repo.html_url = "https://github.com/o/r"

        ev_issue = _make_incoming_event(
            repo=repo,
            event_type="IssuesEvent",
            event_id="e1",
            actor_login="alice",
            created_at="2020-01-01T00:00:00Z",
            payload={"action": "opened", "issue": {"number": 5, "title": "Bug"}},
        )
        ev_comment = _make_incoming_event(
            repo=repo,
            event_type="IssueCommentEvent",
            event_id="e2",
            actor_login="bob",
            created_at="2020-01-01T01:00:00Z",
            payload={"comment": {"body": "agreed"}, "issue": {"number": 5}},
        )
        assert ev_issue.session_key == ev_comment.session_key

    def test_push_event_uses_per_event_key(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        repo = MagicMock()
        repo.id = 42
        repo.full_name = "o/r"
        repo.clone_url = "https://github.com/o/r.git"
        repo.default_branch = "main"
        repo.html_url = "https://github.com/o/r"

        ev = _make_incoming_event(
            repo=repo,
            event_type="PushEvent",
            event_id="e99",
            actor_login="alice",
            created_at="2020-01-01T00:00:00Z",
            payload={"ref": "refs/heads/main", "commits": []},
        )
        assert ev.session_key == SessionKey("github/42/pushevent/e99")

    def test_project_name_in_session_key(self) -> None:
        """When project_name is provided, the session key uses it."""
        from thorn.gateway.sources._github import _make_incoming_event

        repo = MagicMock()
        repo.id = 42
        repo.full_name = "o/r"
        repo.clone_url = "https://github.com/o/r.git"
        repo.default_branch = "main"
        repo.html_url = "https://github.com/o/r"

        ev = _make_incoming_event(
            repo=repo,
            event_type="IssuesEvent",
            event_id="e1",
            actor_login="alice",
            created_at="2020-01-01T00:00:00Z",
            payload={"action": "opened", "issue": {"number": 5, "title": "Bug"}},
            project_name="my-proj",
        )
        assert ev.session_key == SessionKey("my-proj/issue/5")
        assert ev.metadata["project_name"] == "my-proj"

    def test_empty_project_name_uses_legacy_key(self) -> None:
        """When project_name is empty, falls back to repo-id-based key."""
        from thorn.gateway.sources._github import _make_incoming_event

        repo = MagicMock()
        repo.id = 42
        repo.full_name = "o/r"
        repo.clone_url = "https://github.com/o/r.git"
        repo.default_branch = "main"
        repo.html_url = "https://github.com/o/r"

        ev = _make_incoming_event(
            repo=repo,
            event_type="IssuesEvent",
            event_id="e1",
            actor_login="alice",
            created_at="2020-01-01T00:00:00Z",
            payload={"action": "opened", "issue": {"number": 5, "title": "Bug"}},
        )
        assert ev.session_key == SessionKey("github/42/issue/5")


# ---------------------------------------------------------------------------
# GitHubNotificationsSource (mocked)
# ---------------------------------------------------------------------------


def _make_mock_github_event(
    *,
    eid: str,
    event_type: str = "IssuesEvent",
    payload: dict[str, object] | None = None,
    actor_login: str = "bob",
) -> MagicMock:
    ev = MagicMock()
    ev.id = eid
    ev.type = event_type
    ev.payload = payload or {"action": "opened", "issue": {"number": 1, "title": "T"}}
    actor = MagicMock()
    actor.login = actor_login
    ev.actor = actor
    ev.created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return ev


class TestGitHubNotificationsSourceFetchNewEvents:
    """Exercise :meth:`GitHubNotificationsSource._fetch_new_events` directly."""

    def test_first_fetch_primes_and_returns_empty(self) -> None:
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
        ):
            mock_repo = MagicMock()
            mock_repo.id = 456
            mock_repo.full_name = "octocat/hello-world"
            mock_repo.clone_url = "https://github.com/octocat/hello-world.git"
            mock_repo.default_branch = "main"
            mock_repo.html_url = "https://github.com/octocat/hello-world"
            ev_a = _make_mock_github_event(eid="111")
            mock_repo.get_events.return_value = [ev_a]

            mock_gh = MagicMock()
            mock_gh.get_repo.return_value = mock_repo
            mock_gh_cls.return_value = mock_gh

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
            )
            source = GitHubNotificationsSource(config)

            assert source._fetch_new_events() == []
            assert source._primed is True
            assert "111" in source._seen_event_ids

    def test_subsequent_fetch_returns_only_new_ids(self) -> None:
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
        ):
            mock_repo = MagicMock()
            mock_repo.id = 456
            mock_repo.full_name = "octocat/hello-world"
            mock_repo.clone_url = "https://github.com/octocat/hello-world.git"
            mock_repo.default_branch = "main"
            mock_repo.html_url = "https://github.com/octocat/hello-world"

            ev_old = _make_mock_github_event(eid="111")
            ev_new = _make_mock_github_event(eid="222", event_type="IssueCommentEvent")

            mock_repo.get_events.side_effect = [
                [ev_old],
                [ev_new, ev_old],
            ]

            mock_gh = MagicMock()
            mock_gh.get_repo.return_value = mock_repo
            mock_gh_cls.return_value = mock_gh

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
            )
            source = GitHubNotificationsSource(config)

            assert source._fetch_new_events() == []
            second = source._fetch_new_events()
            assert len(second) == 1
            assert second[0].metadata["event_id"] == "222"
            assert second[0].source == "github"


class TestGitHubNotificationsSourceStart:
    @pytest.mark.asyncio
    async def test_start_invokes_poll_loop(self) -> None:
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
        ):
            mock_repo = MagicMock()
            mock_repo.id = 1
            mock_repo.full_name = "o/r"
            mock_repo.clone_url = "https://github.com/o/r.git"
            mock_repo.default_branch = "main"
            mock_repo.html_url = "https://github.com/o/r"
            mock_repo.get_events.return_value = []

            mock_gh = MagicMock()
            mock_gh.get_repo.return_value = mock_repo
            mock_gh_cls.return_value = mock_gh

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="o/r",
                poll_interval=5,
            )
            source = GitHubNotificationsSource(config)

            async def on_event(_event: IncomingEvent) -> None:
                pass

            task = asyncio.create_task(source.start(on_event))
            await asyncio.sleep(0.05)
            await source.stop()
            await asyncio.wait_for(task, timeout=2.0)

            assert mock_gh.get_repo.called


# ---------------------------------------------------------------------------
# GitHubNotificationsSourceConfig
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("github_pat_only_env", "clear_github_api_url_env")
class TestGitHubNotificationsSourceConfig:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github"),
        ):
            from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

            monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
            monkeypatch.setenv("GITHUB_URL", "https://gh.corp.example.com")
            monkeypatch.setenv("THORN_GITHUB_REPOSITORY", "org/repo")
            monkeypatch.setenv("THORN_GITHUB_APP_SLUG", "my-bot")
            monkeypatch.setenv("THORN_POLL_INTERVAL", "15")

            config = GitHubNotificationsSourceConfig.from_env()
            assert config.auth.kind == "pat"
            assert config.auth.token == "ghp_secret"
            assert config.base_url == "https://gh.corp.example.com"
            assert config.repository == "org/repo"
            assert config.app_slug == "my-bot"
            assert config.poll_interval == 15

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github"),
        ):
            from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

            monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
            monkeypatch.setenv("THORN_GITHUB_REPOSITORY", "org/repo")
            monkeypatch.delenv("GITHUB_URL", raising=False)
            monkeypatch.delenv("THORN_GITHUB_APP_SLUG", raising=False)
            monkeypatch.delenv("THORN_POLL_INTERVAL", raising=False)

            config = GitHubNotificationsSourceConfig.from_env()
            assert config.base_url == "https://api.github.com"
            assert config.app_slug == ""
            assert config.poll_interval == 30

    def test_from_env_missing_token_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("THORN_GITHUB_REPOSITORY", "org/repo")
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            GitHubNotificationsSourceConfig.from_env()

    def test_from_env_missing_repository_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.delenv("THORN_GITHUB_REPOSITORY", raising=False)
        with pytest.raises(ValueError, match="THORN_GITHUB_REPOSITORY"):
            GitHubNotificationsSourceConfig.from_env()


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


class TestGitHubSourceRegistry:
    def test_github_registered(self) -> None:
        from thorn.gateway.sources import get_registered_source
        from thorn.gateway.sources._github import GitHubNotificationsSource

        assert get_registered_source("github") is GitHubNotificationsSource

    def test_github_source_has_config_attribute(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        assert GitHubNotificationsSource.Config is GitHubNotificationsSourceConfig


# ---------------------------------------------------------------------------
# instantiate_sources integration
# ---------------------------------------------------------------------------


class TestGitHubInstantiateSources:
    def test_instantiates_github_event_source(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github"),
        ):
            from thorn.gateway._config import (
                GatewayConfig,
                ServiceSpec,
                instantiate_sources,
            )
            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
            )

            monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

            config = GatewayConfig(services=[
                ServiceSpec(
                    name="test-gh",
                    type="github-events",
                    config={
                        "auth": {
                            "kind": "pat",
                            "token": "$GITHUB_TOKEN",
                        },
                        "repository": "owner/repo",
                    },
                ),
            ])
            sources = instantiate_sources(config)
            assert len(sources) == 1
            assert isinstance(sources[0], GitHubNotificationsSource)
