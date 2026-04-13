"""Tests for GitHubNotificationsSource and related helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from thorn.gateway._event import EventSource, IncomingEvent
from thorn.runtime import SessionKey
from thorn.tools._github_connection import GitHubPatAuth


# ---------------------------------------------------------------------------
# Mock notification builder
# ---------------------------------------------------------------------------


def _make_mock_notification(
    thread_id: str = "100",
    repo_id: int = 456,
    repo_full_name: str = "octocat/hello-world",
    subject_type: str = "Issue",
    subject_number: int = 42,
    reason: str = "mention",
    clone_url: str = "https://github.com/octocat/hello-world.git",
    default_branch: str = "main",
    html_url: str = "https://github.com/octocat/hello-world",
    latest_comment_url: str | None = None,
) -> MagicMock:
    """Build a mock that looks like a PyGithub ``Notification``."""
    notification = MagicMock()
    notification.id = thread_id

    repo = MagicMock()
    repo.id = repo_id
    repo.full_name = repo_full_name
    repo.clone_url = clone_url
    repo.default_branch = default_branch
    repo.html_url = html_url
    notification.repository = repo

    subject = MagicMock()
    subject.type = subject_type
    base_url = "https://api.github.com/repos"
    endpoint = "pulls" if subject_type == "PullRequest" else "issues"
    subject.url = f"{base_url}/{repo_full_name}/{endpoint}/{subject_number}"
    subject.latest_comment_url = latest_comment_url
    subject.title = f"Test {subject_type} #{subject_number}"
    notification.subject = subject

    notification.reason = reason
    notification.unread = True
    return notification


# ---------------------------------------------------------------------------
# _extract_subject_number
# ---------------------------------------------------------------------------


class TestExtractSubjectNumber:
    def test_issue_url(self):
        from thorn.gateway.sources._github import _extract_subject_number

        url = "https://api.github.com/repos/octocat/hello/issues/42"
        assert _extract_subject_number(url) == 42

    def test_pull_url(self):
        from thorn.gateway.sources._github import _extract_subject_number

        url = "https://api.github.com/repos/octocat/hello/pulls/7"
        assert _extract_subject_number(url) == 7

    def test_trailing_slash_stripped(self):
        from thorn.gateway.sources._github import _extract_subject_number

        url = "https://api.github.com/repos/octocat/hello/issues/42/"
        assert _extract_subject_number(url) == 42

    def test_non_numeric_returns_none(self):
        from thorn.gateway.sources._github import _extract_subject_number

        url = "https://api.github.com/repos/octocat/hello/commits/abc123"
        assert _extract_subject_number(url) is None

    def test_empty_string_returns_none(self):
        from thorn.gateway.sources._github import _extract_subject_number

        assert _extract_subject_number("") is None


# ---------------------------------------------------------------------------
# Event formatting
# ---------------------------------------------------------------------------


class TestGitHubSourceEventFormatting:
    def test_make_session_key(self):
        from thorn.gateway.sources._github import _make_session_key

        notification = _make_mock_notification(
            repo_id=456, subject_type="PullRequest", subject_number=7,
        )
        key = _make_session_key(notification)
        assert key == SessionKey("github_456_PullRequest_7")
        assert isinstance(key, SessionKey)

    def test_session_key_is_filesystem_safe(self):
        from thorn.gateway.sources._github import _make_session_key

        notification = _make_mock_notification()
        key = _make_session_key(notification)
        forbidden = set('/:*?"<>|\\')
        assert not any(c in forbidden for c in str(key))

    def test_same_subject_different_thread_ids_share_session_key(self):
        """Two notifications about the same issue (different thread IDs)
        produce identical session keys -- this is how multi-turn on a
        single issue or PR works."""
        from thorn.gateway.sources._github import _make_session_key

        notif_a = _make_mock_notification(
            thread_id="100", repo_id=42, subject_type="Issue",
            subject_number=7,
        )
        notif_b = _make_mock_notification(
            thread_id="200", repo_id=42, subject_type="Issue",
            subject_number=7,
        )
        assert _make_session_key(notif_a) == _make_session_key(notif_b)

    def test_issue_and_pr_produce_different_session_keys(self):
        """An Issue notification and a PullRequest notification on the
        same repo produce distinct session keys, even with the same
        number."""
        from thorn.gateway.sources._github import _make_session_key

        issue_notif = _make_mock_notification(
            thread_id="1", repo_id=42, subject_type="Issue",
            subject_number=7,
        )
        pr_notif = _make_mock_notification(
            thread_id="2", repo_id=42, subject_type="PullRequest",
            subject_number=7,
        )
        assert _make_session_key(issue_notif) != _make_session_key(pr_notif)

    def test_make_event(self):
        from thorn.gateway.sources._github import _make_event

        notification = _make_mock_notification(
            thread_id="99",
            repo_id=456,
            repo_full_name="octocat/hello-world",
            subject_type="Issue",
            subject_number=42,
            reason="mention",
        )
        event = _make_event(notification, comment_body="Please fix this")

        assert event.source == "github"
        assert event.session_key == SessionKey("github_456_Issue_42")
        assert "mention" in event.content
        assert "Issue #42" in event.content
        assert "Please fix this" in event.content
        assert "forge_mark_notification_done" in event.content
        assert event.metadata["thread_id"] == "99"
        assert event.metadata["repo_id"] == 456
        assert event.metadata["repo_full_name"] == "octocat/hello-world"
        assert event.metadata["subject_type"] == "Issue"
        assert event.metadata["subject_number"] == 42
        assert event.metadata["reason"] == "mention"
        assert event.metadata["clone_url"] == (
            "https://github.com/octocat/hello-world.git"
        )
        assert event.metadata["default_branch"] == "main"

    def test_make_event_without_comment_body(self):
        from thorn.gateway.sources._github import _make_event

        notification = _make_mock_notification()
        event = _make_event(notification, comment_body=None)

        assert event.source == "github"
        assert "Comment body:" not in event.content
        assert "forge_mark_notification_done" in event.content

    def test_format_event_content_includes_repo_info(self):
        from thorn.gateway.sources._github import _format_event_content

        notification = _make_mock_notification()
        content = _format_event_content(notification, comment_body=None)
        assert "forge_mark_notification_done" in content
        assert "Clone URL:" in content
        assert "Default branch:" in content
        assert "Repository URL:" in content
        assert "Thread ID:" in content
        assert "Reason:" in content

    def test_format_event_content_includes_comment(self):
        from thorn.gateway.sources._github import _format_event_content

        notification = _make_mock_notification()
        content = _format_event_content(
            notification, comment_body="Hey bot, fix the tests!",
        )
        assert "Comment body:" in content
        assert "Hey bot, fix the tests!" in content

    def test_session_key_fallback_when_no_number(self):
        """When the subject URL doesn't contain a numeric tail, the
        session key falls back to using the thread ID."""
        from thorn.gateway.sources._github import _make_session_key

        notification = _make_mock_notification(thread_id="777")
        notification.subject.url = (
            "https://api.github.com/repos/o/r/commits/abc123"
        )
        key = _make_session_key(notification)
        assert "777" in str(key)


# ---------------------------------------------------------------------------
# _fetch_body
# ---------------------------------------------------------------------------


class TestFetchBody:
    def test_returns_body_on_success(self):
        from thorn.gateway.sources._github import _fetch_body

        with patch("thorn.gateway.sources._github.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = {"body": "Hello world"}
            mock_response.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_response

            result = _fetch_body("https://api.example.com/thing/1", "tok")
            assert result == "Hello world"
            mock_httpx.get.assert_called_once()
            call_kwargs = mock_httpx.get.call_args
            assert "Bearer tok" in str(call_kwargs)

    def test_returns_none_on_error(self):
        from thorn.gateway.sources._github import _fetch_body

        with patch("thorn.gateway.sources._github.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("network error")
            result = _fetch_body("https://api.example.com/thing/1", "tok")
            assert result is None

    def test_returns_none_on_empty_body(self):
        from thorn.gateway.sources._github import _fetch_body

        with patch("thorn.gateway.sources._github.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = {"body": ""}
            mock_response.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_response

            result = _fetch_body("https://api.example.com/thing/1", "tok")
            assert result is None


# ---------------------------------------------------------------------------
# GitHubNotificationsSource (mocked)
# ---------------------------------------------------------------------------


class TestGitHubNotificationsSourcePolling:
    @pytest.mark.asyncio
    async def test_polls_and_emits_events(self):
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
        ):
            mock_gh = MagicMock()
            mock_gh_cls.return_value = mock_gh

            mock_user = MagicMock()
            mock_user.login = "thorn-app[bot]"
            mock_user.name = "Thorn App"
            mock_user.html_url = "https://github.com/apps/thorn-app"
            mock_gh.get_user.return_value = mock_user

            notification = _make_mock_notification()
            mock_user.get_notifications.return_value = [notification]

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
                poll_interval=5,
            )
            source = GitHubNotificationsSource(config)

            events: list[IncomingEvent] = []

            async def on_event(event: IncomingEvent) -> None:
                events.append(event)
                await source.stop()

            with patch.object(source, "_fetch_comment_body", return_value=None):
                await asyncio.wait_for(
                    source.start(on_event), timeout=5.0,
                )

            assert len(events) == 1
            assert events[0].source == "github"
            assert events[0].session_key == SessionKey(
                "github_456_Issue_42",
            )

    @pytest.mark.asyncio
    async def test_deduplicates_notifications(self):
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
        ):
            mock_gh = MagicMock()
            mock_gh_cls.return_value = mock_gh

            mock_user = MagicMock()
            mock_user.login = "bot"
            mock_user.name = "Bot"
            mock_user.html_url = "https://github.com/bot"
            mock_gh.get_user.return_value = mock_user

            notification = _make_mock_notification(thread_id="42")
            mock_user.get_notifications.return_value = [notification]

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
                poll_interval=5,
            )
            source = GitHubNotificationsSource(config)

            events: list[IncomingEvent] = []

            async def on_event(event: IncomingEvent) -> None:
                events.append(event)

            with patch.object(source, "_fetch_comment_body", return_value=None):
                await source._poll_once(on_event)
                await source._poll_once(on_event)

            assert len(events) == 1

    @pytest.mark.asyncio
    async def test_emits_both_notifications_with_same_session_key(self):
        """Two notifications with different IDs but the same subject
        (same session key) should both be emitted.  _seen deduplicates
        by notification ID, not by session key."""
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
        ):
            mock_gh = MagicMock()
            mock_gh_cls.return_value = mock_gh

            mock_user = MagicMock()
            mock_user.login = "bot"
            mock_user.name = "Bot"
            mock_user.html_url = "https://github.com/bot"
            mock_gh.get_user.return_value = mock_user

            notif_a = _make_mock_notification(
                thread_id="10", subject_type="Issue", subject_number=7,
            )
            notif_b = _make_mock_notification(
                thread_id="20", subject_type="Issue", subject_number=7,
            )
            mock_user.get_notifications.return_value = [notif_a, notif_b]

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
                poll_interval=5,
            )
            source = GitHubNotificationsSource(config)

            events: list[IncomingEvent] = []

            async def on_event(event: IncomingEvent) -> None:
                events.append(event)

            with patch.object(source, "_fetch_comment_body", return_value=None):
                await source._poll_once(on_event)

            assert len(events) == 2
            assert events[0].session_key == events[1].session_key
            assert (
                events[0].metadata["thread_id"]
                != events[1].metadata["thread_id"]
            )

    @pytest.mark.asyncio
    async def test_filters_by_repository(self):
        """Notifications from other repos are filtered out."""
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
        ):
            mock_gh = MagicMock()
            mock_gh_cls.return_value = mock_gh

            mock_user = MagicMock()
            mock_user.login = "bot"
            mock_user.name = "Bot"
            mock_user.html_url = "https://github.com/bot"
            mock_gh.get_user.return_value = mock_user

            wanted = _make_mock_notification(
                thread_id="1",
                repo_full_name="octocat/hello-world",
            )
            unwanted = _make_mock_notification(
                thread_id="2",
                repo_full_name="octocat/other-repo",
            )
            mock_user.get_notifications.return_value = [wanted, unwanted]

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
                poll_interval=5,
            )
            source = GitHubNotificationsSource(config)

            events: list[IncomingEvent] = []

            async def on_event(event: IncomingEvent) -> None:
                events.append(event)

            with patch.object(source, "_fetch_comment_body", return_value=None):
                await source._poll_once(on_event)

            assert len(events) == 1
            assert events[0].metadata["repo_full_name"] == (
                "octocat/hello-world"
            )

    @pytest.mark.asyncio
    async def test_fetches_comment_body(self):
        """_poll_once should call _fetch_comment_body for each new
        notification and include the result in the event."""
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
        ):
            mock_gh = MagicMock()
            mock_gh_cls.return_value = mock_gh

            mock_user = MagicMock()
            mock_user.login = "bot"
            mock_user.name = "Bot"
            mock_user.html_url = "https://github.com/bot"
            mock_gh.get_user.return_value = mock_user

            notification = _make_mock_notification()
            mock_user.get_notifications.return_value = [notification]

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
                poll_interval=5,
            )
            source = GitHubNotificationsSource(config)

            events: list[IncomingEvent] = []

            async def on_event(event: IncomingEvent) -> None:
                events.append(event)

            with patch.object(
                source,
                "_fetch_comment_body",
                return_value="@thorn-app please fix this",
            ):
                await source._poll_once(on_event)

            assert len(events) == 1
            assert "@thorn-app please fix this" in events[0].content


class TestGitHubNotificationsSourceFetchCommentBody:
    """Test the _fetch_comment_body method's fallback logic."""

    def test_prefers_latest_comment_url(self):
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
            patch("thorn.gateway.sources._github._fetch_body") as mock_fetch,
        ):
            mock_gh_cls.return_value = MagicMock()

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
            )
            source = GitHubNotificationsSource(config)

            notification = _make_mock_notification(
                latest_comment_url=(
                    "https://api.github.com/repos/o/r/issues/comments/999"
                ),
            )
            mock_fetch.return_value = "comment body"

            result = source._fetch_comment_body(notification)

            assert result == "comment body"
            mock_fetch.assert_called_once_with(
                "https://api.github.com/repos/o/r/issues/comments/999",
                "ghp_test",
            )

    def test_falls_back_to_subject_url(self):
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
            patch("thorn.gateway.sources._github._fetch_body") as mock_fetch,
        ):
            mock_gh_cls.return_value = MagicMock()

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
            )
            source = GitHubNotificationsSource(config)

            notification = _make_mock_notification()
            notification.subject.latest_comment_url = None

            mock_fetch.side_effect = lambda url, token: (
                "issue body"
                if "issues/42" in url
                else None
            )

            result = source._fetch_comment_body(notification)

            assert result == "issue body"

    def test_returns_none_when_all_fail(self):
        with (
            patch("thorn.gateway.sources._github._HAS_GITHUB", True),
            patch("thorn.gateway.sources._github._Github") as mock_gh_cls,
            patch("thorn.gateway.sources._github._fetch_body") as mock_fetch,
        ):
            mock_gh_cls.return_value = MagicMock()

            from thorn.gateway.sources._github import (
                GitHubNotificationsSource,
                GitHubNotificationsSourceConfig,
            )

            config = GitHubNotificationsSourceConfig(
                auth=GitHubPatAuth(token="ghp_test"),
                repository="octocat/hello-world",
            )
            source = GitHubNotificationsSource(config)

            notification = _make_mock_notification()
            notification.subject.latest_comment_url = None
            mock_fetch.return_value = None

            result = source._fetch_comment_body(notification)
            assert result is None


# ---------------------------------------------------------------------------
# GitHubNotificationsSourceConfig
# ---------------------------------------------------------------------------


class TestGitHubNotificationsSourceConfig:
    def test_from_env(
        self, monkeypatch: pytest.MonkeyPatch, github_pat_only_env: None,
    ):
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

    def test_from_env_defaults(
        self, monkeypatch: pytest.MonkeyPatch, github_pat_only_env: None,
    ):
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
        self, monkeypatch: pytest.MonkeyPatch, github_pat_only_env: None,
    ):
        from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("THORN_GITHUB_REPOSITORY", "org/repo")
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            GitHubNotificationsSourceConfig.from_env()

    def test_from_env_missing_repository_raises(
        self, monkeypatch: pytest.MonkeyPatch, github_pat_only_env: None,
    ):
        from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.delenv("THORN_GITHUB_REPOSITORY", raising=False)
        with pytest.raises(ValueError, match="THORN_GITHUB_REPOSITORY"):
            GitHubNotificationsSourceConfig.from_env()


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


class TestGitHubSourceRegistry:
    def test_github_registered(self):
        from thorn.gateway.sources import get_registered_source
        from thorn.gateway.sources._github import GitHubNotificationsSource

        assert get_registered_source("github") is GitHubNotificationsSource

    def test_github_source_has_config_attribute(self):
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
    ):
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
