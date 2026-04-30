"""Tests for GitHubNotificationsSource (Notifications API)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from thorn.gateway._event import EventKind, RawIncomingEvent
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


def _comment_payload(
    *,
    body: str = "comment text",
    user_id: int | None = 12345,
    user_login: str = "stranger",
    user_type: str | None = "User",
    created_at: str = "2026-04-15T09:30:00Z",
) -> dict[str, Any]:
    """Build a minimal GitHub comment / issue payload for tests."""
    payload: dict[str, Any] = {"body": body, "created_at": created_at}
    if user_id is not None or user_login or user_type is not None:
        user: dict[str, Any] = {}
        if user_id is not None:
            user["id"] = user_id
        if user_login:
            user["login"] = user_login
        if user_type is not None:
            user["type"] = user_type
        payload["user"] = user
    return payload


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
# _make_raw_event
# ---------------------------------------------------------------------------


class TestMakeRawEvent:
    def test_metadata(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread()
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(
                body="Please fix this",
                user={"id": 7, "login": "stranger", "type": "User"},
                created_at="2026-04-15T09:00:00Z",
            ),
            native_id_to_project_name={},
            base_url="https://api.github.com",
        )
        assert ev.source == "github"
        assert ev.kind is EventKind.CONVERSATIONAL  # reason="mention"
        assert ev.metadata["notification_id"] == "100"
        assert ev.metadata["reason"] == "mention"
        assert ev.metadata["repo_id"] == 456
        assert ev.metadata["repo_full_name"] == "octocat/hello-world"
        # The summary carries harness-controlled prose; the body
        # is *not* in the summary -- that is in items[0].
        assert "Please fix this" not in ev.summary
        assert "marked read on your behalf" in ev.summary
        # Body lives on the first context item.
        assert len(ev.items) == 1
        assert ev.items[0].body == "Please fix this"
        # Source-namespaced external_key keyed on the API base URL,
        # the notification thread id, *and* the thread's
        # ``updated_at`` snapshot.  Including ``updated_at`` ensures
        # that two genuinely distinct events on the same thread
        # (which share a stable thread ID but have different
        # ``updated_at`` values) are treated as distinct logical
        # notifications.
        assert ev.external_key == (
            "github:https://api.github.com:thread:100:updated:2026-04-15T10:00:00Z"
        )

    def test_actor_captured_from_comment(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread()
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(
                body="hi",
                user={"id": 12345, "login": "stranger", "type": "User"},
                created_at="2026-04-15T09:00:00Z",
            ),
            native_id_to_project_name={},
            forge_name="github",
        )
        actor = ev.primary_actor
        assert actor is not None
        assert actor.service == "github"
        assert actor.account_id == "12345"
        assert "stranger" in actor.secondary_account_ids
        assert actor.is_bot is False

    def test_bot_actor_is_flagged(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread()
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(
                body="bleep",
                user={"id": 99, "login": "dependabot[bot]", "type": "Bot"},
                created_at="",
            ),
            native_id_to_project_name={},
        )
        assert ev.primary_actor is not None
        assert ev.primary_actor.is_bot is True

    def test_no_user_yields_none_actor(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread()
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(body="no user here", user=None),
            native_id_to_project_name={},
        )
        assert ev.primary_actor is None

    def test_reason_subscribed_classifies_as_structural(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread(reason="subscribed")
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(),
            native_id_to_project_name={},
        )
        assert ev.kind is EventKind.STRUCTURAL

    def test_issue_routes_to_issue_session(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread(repo_id=42)
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(),
            native_id_to_project_name={},
        )
        assert ev.session_key == SessionKey("github/42/issue/7")

    def test_pull_request_routes_to_change_request_session(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread(
            repo_id=42,
            subject_type="PullRequest",
            subject_url="https://api.github.com/repos/o/r/pulls/3",
        )
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(),
            native_id_to_project_name={},
        )
        assert ev.session_key == SessionKey("github/42/change-request/3")

    def test_commit_uses_per_thread_key(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread(
            thread_id="t999",
            repo_id=42,
            subject_type="Commit",
            subject_url="https://api.github.com/repos/o/r/commits/abc",
        )
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(),
            native_id_to_project_name={},
        )
        assert ev.session_key == SessionKey("github/42/commit/t999")

    def test_project_name_in_session_key(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread(repo_id=42)
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(),
            native_id_to_project_name={"octocat/hello-world": "my-proj"},
        )
        assert ev.session_key == SessionKey("my-proj/issue/7")
        assert ev.metadata["project_name"] == "my-proj"

    def test_empty_project_name_uses_legacy_key(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread = _make_notification_thread(repo_id=42)
        ev = _make_raw_event(
            thread=thread,
            comment_info=_LatestCommentInfo(),
            native_id_to_project_name={},
        )
        assert ev.session_key == SessionKey("github/42/issue/7")


# ---------------------------------------------------------------------------
# GitHubNotificationsSource (mocked)
# ---------------------------------------------------------------------------


class TestGitHubNotificationsSourceFetchNewNotifications:
    """Exercise :meth:`GitHubNotificationsSource._fetch_new_notifications` directly.

    The source no longer maintains client-side dedup state -- it
    returns one ``RawIncomingEvent`` per currently-unread thread on
    every poll.  "Already processed" is tracked server-side via the
    Notifications API's read/unread bit (see the startup-drain test
    below and the ``_poll_once`` mark-read tests).
    """

    def test_fetch_emits_event_for_each_returned_thread(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(token="ghp_test")
        source = GitHubNotificationsSource(config)

        thread_a = _make_notification_thread(
            thread_id="111", updated_at="2026-04-15T10:00:00Z",
        )
        thread_b = _make_notification_thread(
            thread_id="222",
            reason="assign",
            updated_at="2026-04-15T11:00:00Z",
        )

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            if "/issues/comments/" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = _comment_payload()
                return resp

            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.headers = {}
            # Return them in non-chronological order so the test also
            # exercises the chronological sort.
            resp.json.return_value = [thread_b, thread_a]
            return resp

        with patch.object(source._http, "get", side_effect=mock_get):
            result = source._fetch_new_notifications()

        assert [ev.metadata["notification_id"] for ev in result] == ["111", "222"]
        assert all(ev.source == "github" for ev in result)

    def test_fetch_returns_empty_on_304(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(token="ghp_test")
        source = GitHubNotificationsSource(config)

        mock_resp = MagicMock()
        mock_resp.status_code = 304
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {}

        with patch.object(source._http, "get", return_value=mock_resp):
            assert source._fetch_new_notifications() == []


class TestGitHubNotificationsSourceDrainExistingUnread:
    """Exercise the startup-drain step."""

    def test_drain_marks_each_unread_thread_read_without_emitting(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(token="ghp_test")
        source = GitHubNotificationsSource(config)

        thread_a = _make_notification_thread(thread_id="111")
        thread_b = _make_notification_thread(thread_id="222", reason="assign")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = [thread_a, thread_b]
        list_resp.headers = {}

        patch_resp = MagicMock()
        patch_resp.status_code = 205
        patch_resp.raise_for_status = MagicMock()

        patch_calls: list[str] = []

        def mock_patch(url: str, **kwargs: Any) -> MagicMock:
            patch_calls.append(url)
            return patch_resp

        with (
            patch.object(source._http, "get", return_value=list_resp),
            patch.object(source._http, "patch", side_effect=mock_patch),
        ):
            count = source._drain_existing_unread()

        assert count == 2
        assert sorted(patch_calls) == [
            "/notifications/threads/111",
            "/notifications/threads/222",
        ]
        # The drain must clear ``_last_modified`` so the first
        # steady-state poll isn't short-circuited to 304 against
        # the snapshot it just consumed.
        assert source._last_modified is None

    def test_new_activity_on_previously_drained_thread_is_delivered(self) -> None:
        """Regression test: the original bug.

        A thread that was unread at startup (and so got marked read
        in the drain) must still produce an event on the next poll
        if new activity flips it back to unread (with a fresh
        ``updated_at``).  Under the old client-side seen-set this
        new activity was silently dropped.
        """
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        config = GitHubNotificationsSourceConfig(token="ghp_test")
        source = GitHubNotificationsSource(config)

        thread_v1 = _make_notification_thread(
            thread_id="111", updated_at="2026-04-15T10:00:00Z",
        )
        thread_v2 = _make_notification_thread(
            thread_id="111", updated_at="2026-04-15T12:00:00Z",
        )

        list_responses = [
            # Drain sees the v1 snapshot.
            [thread_v1],
            # Steady-state poll sees the same thread re-flipped to
            # unread with a newer updated_at.
            [thread_v2],
        ]

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            if "/issues/comments/" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = _comment_payload(body="new comment")
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.headers = {}
            resp.json.return_value = list_responses.pop(0)
            return resp

        patch_resp = MagicMock()
        patch_resp.status_code = 205
        patch_resp.raise_for_status = MagicMock()

        with (
            patch.object(source._http, "get", side_effect=mock_get),
            patch.object(source._http, "patch", return_value=patch_resp),
        ):
            source._drain_existing_unread()
            events = source._fetch_new_notifications()

        assert len(events) == 1
        assert events[0].metadata["notification_id"] == "111"
        assert events[0].metadata["updated_at"] == "2026-04-15T12:00:00Z"
        assert "2026-04-15T12:00:00Z" in events[0].external_key


class TestGitHubExternalKey:
    def test_external_key_includes_updated_at(self) -> None:
        from thorn.gateway.sources._github import (
            _LatestCommentInfo,
            _make_raw_event,
        )

        thread_v1 = _make_notification_thread(
            thread_id="111", updated_at="2026-04-15T10:00:00Z",
        )
        thread_v2 = _make_notification_thread(
            thread_id="111", updated_at="2026-04-15T12:00:00Z",
        )

        ev_v1 = _make_raw_event(
            thread=thread_v1,
            comment_info=_LatestCommentInfo(),
            native_id_to_project_name={},
            base_url="https://api.github.com",
        )
        ev_v2 = _make_raw_event(
            thread=thread_v2,
            comment_info=_LatestCommentInfo(),
            native_id_to_project_name={},
            base_url="https://api.github.com",
        )

        assert ev_v1.external_key != ev_v2.external_key
        # Same thread => same session inbox; the InFlightIndex is the
        # only thing that distinguishes the two versions.
        assert ev_v1.session_key == ev_v2.session_key


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

        thread = _make_notification_thread(thread_id="300")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = [thread]
        list_resp.headers = {}

        comment_resp = MagicMock()
        comment_resp.status_code = 200
        comment_resp.raise_for_status = MagicMock()
        comment_resp.json.return_value = _comment_payload(body="comment")

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

        async def on_event(_event: RawIncomingEvent) -> None:
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

        thread = _make_notification_thread(thread_id="301")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = [thread]
        list_resp.headers = {}

        comment_resp = MagicMock()
        comment_resp.status_code = 200
        comment_resp.raise_for_status = MagicMock()
        comment_resp.json.return_value = _comment_payload(body="comment")

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            if "/issues/comments/" in url:
                return comment_resp
            return list_resp

        patch_calls: list[str] = []

        async def on_event(_event: RawIncomingEvent) -> None:
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

        async def on_event(_event: RawIncomingEvent) -> None:
            pass

        with patch.object(source._http, "get", side_effect=mock_get):
            task = asyncio.create_task(source.start(on_event))
            await asyncio.sleep(0.05)
            await source.stop()
            await asyncio.wait_for(task, timeout=2.0)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


class TestGitHubSourceRegistry:
    def test_github_source_has_config_attribute(self) -> None:
        from thorn.gateway.sources._github import (
            GitHubNotificationsSource,
            GitHubNotificationsSourceConfig,
        )

        assert GitHubNotificationsSource.Config is GitHubNotificationsSourceConfig
