"""Tests for the gateway's per-agent gitconfig rendering.

The gateway writes a tiny ephemeral ``gitconfig`` file for each
agent whose broker binding carries any ``git_extra_headers``
entries, then bind-mounts it read-only into the sandbox container
at a fixed path.  These tests pin the rendered file format and the
``BrokerBinding`` update flow in isolation, without spinning up a
full :class:`~thorn.gateway.Gateway`.
"""

from __future__ import annotations

from thorn.gateway._gateway import _render_git_extra_headers


class TestRenderGitExtraHeaders:
    def test_empty_input_returns_empty_string(self) -> None:
        """No headers → no file contents.  The gateway uses this
        branch to skip writing the file entirely when a binding has
        no git-HTTPS plans."""
        assert _render_git_extra_headers(()) == ""

    def test_single_host_produces_url_scoped_section(self) -> None:
        """Each host gets a ``[http "https://<host>/"]`` section with
        a single ``extraHeader`` value.  Using the URL-scoped
        section shape (rather than a global ``[http]`` block) keeps
        the header scoped to the host we meant to route, while the
        global ``proxyAuthMethod`` entry only affects proxy
        authentication."""
        rendered = _render_git_extra_headers(
            (("github.com", "Authorization: Basic placeholder"),),
        )
        assert rendered == (
            "[http]\n"
            "    proxyAuthMethod = basic\n"
            '[http "https://github.com/"]\n'
            "    extraHeader = Authorization: Basic placeholder\n"
        )

    def test_multiple_hosts_each_get_their_own_section(self) -> None:
        rendered = _render_git_extra_headers((
            ("github.com", "Authorization: Basic ph1"),
            ("gitlab.com", "Authorization: Basic ph2"),
        ))
        assert rendered == (
            "[http]\n"
            "    proxyAuthMethod = basic\n"
            '[http "https://github.com/"]\n'
            "    extraHeader = Authorization: Basic ph1\n"
            '[http "https://gitlab.com/"]\n'
            "    extraHeader = Authorization: Basic ph2\n"
        )

    def test_duplicate_host_first_occurrence_wins(self) -> None:
        """If two plans register the same host, we emit a single
        section.  Emitting two would leave ``git`` silently sending
        two ``Authorization`` headers (last-wins per git's config
        parser semantics), which is harder to reason about than a
        deterministic single entry."""
        rendered = _render_git_extra_headers((
            ("github.com", "Authorization: Basic first"),
            ("github.com", "Authorization: Basic second"),
        ))
        assert rendered.count('[http "https://github.com/"]') == 1
        assert "first" in rendered
        assert "second" not in rendered
