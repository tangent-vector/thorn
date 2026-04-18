"""Tests for GitHubNotificationsSource (Notifications API)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from thorn.gateway._event import IncomingEvent
from thorn.gateway._routing import NoteableKind
from thorn.runtime import SessionKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_notification_thread(
    *,
    thread_id: str = "100",
    repo_id: int = 456,
    repo_full_name: str = "octocat/hello-world",
    subject_type: str = "Issue",
    subject_title: str = "Found a bug",
    subject_url: str = "https://api.github.com/repos/octocat/hello-world/issues/7",
    latest_comment_url: str = "https://api.github.com/repos/octocat/hello-world/issues/comments/999",
    reason: str = "mention",
    updated_at: str = "2026-04-15T10:00:00Z",
    clone_url: str = "https://github.com/octocat/hello-world.git",
    default_branch: str = "main",
    html_url: str = "https://github.com/octocat/hello-world",
) -> dict[str, Any]:
    return {
        "id": thread_id,
        "repository": {
            "id": repo_id,
            "full_name": repo_full_name,
            "clone_url": clone_url,
            "default_branch": default_branch,
            "html_url": html_url,
        },
        "subject": {
            "title": subject_title,
            "url": subject_url,
            "latest_comment_url": latest_comment_url,
            "type": subject_type,
        },
        "reason": reason,
        "unread": True,
        "updated_at": updated_at,
        "last_read_at": None,
        "url": f"https://api.github.com/notifications/threads/{thread_id}",
        "subscription_url": f"https://api.github.com/notifications/threads/{thread_id}/subscription",
    }


# ---------------------------------------------------------------------------
# Session key (routing -- unchanged, just verify it still works)
# ---------------------------------------------------------------------------


class TestSessionKeyForEvent:
    def test_stable_key(self) -> None:
        from thorn.gateway._routing import route_github_event

        k = route_github_event(
            repo_id=42, event_type="IssuesEvent", event_id="evt-123",
        )
        assert k == SessionKey("github/42/issuesevent/evt-123")


# ---------------------------------------------------------------------------
# _extract_noteable_from_notification
# ---------------------------------------------------------------------------


class TestExtractNoteableFromNotification:
    def test_issue(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        n = _extract_noteable_from_notification(
            "Issue", "https://api.github.com/repos/o/r/issues/7",
        )
        assert n is not None
        assert n.kind is NoteableKind.ISSUE
        assert n.number == 7

    def test_pull_request(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        n = _extract_noteable_from_notification(
            "PullRequest", "https://api.github.com/repos/o/r/pulls/10",
        )
        assert n is not None
        assert n.kind is NoteableKind.CHANGE_REQUEST
        assert n.number == 10

    def test_commit_returns_none(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        assert _extract_noteable_from_notification(
            "Commit", "https://api.github.com/repos/o/r/commits/abc",
        ) is None

    def test_release_returns_none(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        assert _extract_noteable_from_notification(
            "Release", "https://api.github.com/repos/o/r/releases/42",
        ) is None

    def test_empty_url_returns_none(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        assert _extract_noteable_from_notification("Issue", "") is None

    def test_url_without_number_returns_none(self) -> None:
        from thorn.gateway.sources._github import _extract_noteable_from_notification

        assert _extract_noteable_from_notification(
            "Issue", "https://api.github.com/repos/o/r/issues",
        ) is None


# ---------------------------------------------------------------------------
# _make_incoming_event
# ---------------------------------------------------------------------------


class TestMakeIncomingEvent:
    def test_metadata(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread()
        ev = _make_incoming_event(
            thread=thread,
            comment_body="Please fix this",
            native_id_to_project_name={},
            base_url="https://api.github.com",
        )
        assert ev.source == "github"
        assert ev.metadata["notification_id"] == "100"
        assert ev.metadata["reason"] == "mention"
        assert ev.metadata["repo_id"] == 456
        assert ev.metadata["repo_full_name"] == "octocat/hello-world"
        assert "Please fix this" in ev.content
        # Source-namespaced external_key keyed on the API base URL
        # and the notification thread id -- this is what the
        # InFlightIndex consults for cross-poll and cross-restart
        # deduplication.
        assert ev.external_key == (
            "github:https://api.github.com:thread:100"
        )
        # The content no longer instructs the agent to close out the
        # notification itself: the source marks the thread read on
        # GitHub at post time.
        assert "forge_mark_notification_done" not in ev.content
        assert "marked read on your behalf" in ev.content

    def test_issue_routes_to_issue_session(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread(repo_id=42)
        ev = _make_incoming_event(
            thread=thread,
            comment_body="",
            native_id_to_project_name={},
        )
        assert ev.session_key == SessionKey("github/42/issue/7")

    def test_pull_request_routes_to_change_request_session(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread(
            repo_id=42,
            subject_type="PullRequest",
            subject_url="https://api.github.com/repos/o/r/pulls/3",
        )
        ev = _make_incoming_event(
            thread=thread,
            comment_body="",
            native_id_to_project_name={},
        )
        assert ev.session_key == SessionKey("github/42/change-request/3")

    def test_commit_uses_per_thread_key(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread(
            thread_id="t999",
            repo_id=42,
            subject_type="Commit",
            subject_url="https://api.github.com/repos/o/r/commits/abc",
        )
        ev = _make_incoming_event(
            thread=thread,
            comment_body="",
            native_id_to_project_name={},
        )
        assert ev.session_key == SessionKey("github/42/commit/t999")

    def test_project_name_in_session_key(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread(repo_id=42)
        ev = _make_incoming_event(
            thread=thread,
            comment_body="",
            native_id_to_project_name={"octocat/hello-world": "my-proj"},
        )
        assert ev.session_key == SessionKey("my-proj/issue/7")
        assert ev.metadata["project_name"] == "my-proj"

    def test_empty_project_name_uses_legacy_key(self) -> None:
        from thorn.gateway.sources._github import _make_incoming_event

        thread = _make_notification_thread(repo_id=42)
        ev = _make_incoming_event(
            thread=thread,
            comment_body="",
            native_id_to_project_name={},
        )
        assert ev.session_key == SessionKey("github/42/issue/7")


# ---------------------------------------------------------------------------
# GitHubNotificationsSource (mocked)
# ---------------------------------------------------------------------------


class TestGitHubNotificationsSourceFetchNewNotifications:
    """Exercise :meth:`GitHubNotificationsSource._fetch_new_notifications` directly."""

    def test_first_fetch_primes_and_returns_empty(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(token="ghp_test")
        source = GitHubNotificationsSource(config)

        thread = _make_notification_thread(thread_id="111")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [thread]
        mock_resp.headers = {}

        with patch.object(source._http, "get", return_value=mock_resp):
            result = source._fetch_new_notifications()

        assert result == []
        assert source._primed is True
        assert "111" in source._seen_thread_ids

    def test_subsequent_fetch_returns_only_new_ids(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(token="ghp_test")
        source = GitHubNotificationsSource(config)

        thread_old = _make_notification_thread(thread_id="111")
        thread_new = _make_notification_thread(thread_id="222", reason="assign")

        call_count = 0

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            if "/issues/comments/" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {"body": "comment text"}
                return resp

            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.headers = {}
            if call_count == 1:
                resp.json.return_value = [thread_old]
            else:
                resp.json.return_value = [thread_new, thread_old]
            return resp

        with patch.object(source._http, "get", side_effect=mock_get):
            assert source._fetch_new_notifications() == []
            second = source._fetch_new_notifications()

        assert len(second) == 1
        assert second[0].metadata["notification_id"] == "222"
        assert second[0].source == "github"


class TestGitHubNotificationsSourcePollOnce:
    """Exercise the mark-read-at-post behaviour of ``_poll_once``."""

    @pytest.mark.asyncio
    async def test_marks_thread_read_after_post(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(token="ghp_test")
        source = GitHubNotificationsSource(config)
        source._primed = True

        thread = _make_notification_thread(thread_id="300")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = [thread]
        list_resp.headers = {}

        comment_resp = MagicMock()
        comment_resp.status_code = 200
        comment_resp.raise_for_status = MagicMock()
        comment_resp.json.return_value = {"body": "comment"}

        patch_resp = MagicMock()
        patch_resp.status_code = 205
        patch_resp.raise_for_status = MagicMock()

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            if "/issues/comments/" in url:
                return comment_resp
            return list_resp

        patch_calls: list[str] = []

        def mock_patch(url: str, **kwargs: Any) -> MagicMock:
            patch_calls.append(url)
            return patch_resp

        async def on_event(_event: IncomingEvent) -> None:
            pass

        with (
            patch.object(source._http, "get", side_effect=mock_get),
            patch.object(source._http, "patch", side_effect=mock_patch),
        ):
            await source._poll_once(on_event)

        assert patch_calls == ["/notifications/threads/300"]

    @pytest.mark.asyncio
    async def test_post_failure_skips_mark_read(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(token="ghp_test")
        source = GitHubNotificationsSource(config)
        source._primed = True

        thread = _make_notification_thread(thread_id="301")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = [thread]
        list_resp.headers = {}

        comment_resp = MagicMock()
        comment_resp.status_code = 200
        comment_resp.raise_for_status = MagicMock()
        comment_resp.json.return_value = {"body": "comment"}

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            if "/issues/comments/" in url:
                return comment_resp
            return list_resp

        patch_calls: list[str] = []

        async def on_event(_event: IncomingEvent) -> None:
            raise RuntimeError("boom")

        with (
            patch.object(source._http, "get", side_effect=mock_get),
            patch.object(
                source._http, "patch",
                side_effect=lambda url, **_: patch_calls.append(url),
            ),
        ):
            await source._poll_once(on_event)

        assert patch_calls == []


class TestGitHubNotificationsSourceStart:
    @pytest.mark.asyncio
    async def test_start_invokes_poll_loop(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(
            token="ghp_test", poll_interval=5,
        )
        source = GitHubNotificationsSource(config)

        mock_user_resp = MagicMock()
        mock_user_resp.status_code = 200
        mock_user_resp.raise_for_status = MagicMock()
        mock_user_resp.json.return_value = {
            "login": "bot", "name": "", "html_url": "",
        }

        mock_notif_resp = MagicMock()
        mock_notif_resp.status_code = 200
        mock_notif_resp.raise_for_status = MagicMock()
        mock_notif_resp.json.return_value = []
        mock_notif_resp.headers = {}

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            if url == "/user":
                return mock_user_resp
            return mock_notif_resp

        async def on_event(_event: IncomingEvent) -> None:
            pass

        with patch.object(source._http, "get", side_effect=mock_get):
            task = asyncio.create_task(source.start(on_event))
            await asyncio.sleep(0.05)
            await source.stop()
            await asyncio.wait_for(task, timeout=2.0)


# ---------------------------------------------------------------------------
# GitHubNotificationsSourceConfig
# ---------------------------------------------------------------------------


class TestGitHubNotificationsSourceConfig:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_API_URL", "https://gh.corp.example.com")
        monkeypatch.setenv("THORN_POLL_INTERVAL", "15")

        config = GitHubNotificationsSourceConfig.from_env()
        assert config.token == "ghp_secret"
        assert config.base_url == "https://gh.corp.example.com"
        assert config.poll_interval == 15

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.delenv("GITHUB_API_URL", raising=False)
        monkeypatch.delenv("THORN_POLL_INTERVAL", raising=False)

        config = GitHubNotificationsSourceConfig.from_env()
        assert config.base_url == "https://api.github.com"
        assert config.poll_interval == 30

    def test_from_env_missing_token_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from thorn.gateway.sources._github import GitHubNotificationsSourceConfig

        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
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
# instantiate_sources integration (legacy format)
# ---------------------------------------------------------------------------


class TestGitHubInstantiateSources:
    def test_instantiates_github_event_source(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
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
                    "token": "$GITHUB_TOKEN",
                },
            ),
        ])
        sources = instantiate_sources(config)
        assert len(sources) == 1
        assert isinstance(sources[0], GitHubNotificationsSource)
