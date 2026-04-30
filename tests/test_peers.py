"""Tests for the peer-identity-and-trust feature set.

Covers the new types/policies introduced for peer-aware event
filtering and content envelope wrapping:

- :class:`PeerSpec` validation grammar.
- :class:`PeerRegistry` lookup (immutable id, secondary handle,
  ``find_by_name``, ``lookup_account``, ``all_peers``,
  duplicate-id and account-collision errors).
- :func:`wrap_external` envelope rendering.
- :class:`TriggerAuthorizationPolicy` decisions for every branch
  of the lattice (system, peer-conv, non-peer-conv,
  peer-structural, non-peer-structural with carve-out on/off,
  bot-default-deny, bot-as-peer).
- :class:`NotificationFormatter` end-to-end behaviour (rendered
  ``content`` shape, banner insertion, drop reasons).
- ``thorn.tools.peers`` agent-facing tools
  (``peer_by_account``, ``find_peers_by_name``, ``list_peers``).

The forge-tool envelope wrapping has its own coverage in
``test_tools.py``-style integration tests; here we stick to the
gateway-internal types and the dedicated peer tools.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from thorn.core._agent import Agent
from thorn.core._context import (
    ExecutionContext,
    reset_context,
    set_context,
)
from thorn.core._provider import MockProvider
from thorn.gateway._actor import ActorIdentity, describe_actor_for_log
from thorn.gateway._envelope import PeerStatus, wrap_external
from thorn.gateway._event import (
    ContextItem,
    ContextItemKind,
    EventKind,
    RawIncomingEvent,
)
from thorn.gateway._formatter import (
    FormatterDelivery,
    FormatterDrop,
    NotificationFormatter,
)
from thorn.gateway._peer import (
    PeerAccount,
    PeerKind,
    PeerRegistry,
    PeerSpec,
)
from thorn.gateway._trigger_policy import (
    Deliver,
    DeliverWithBanner,
    Drop,
    SourceTriggerPolicy,
    TriggerAuthorizationPolicy,
)
from thorn.runtime import AgentID, Runtime, SessionKey
from thorn.tools.peers import (
    Peer,
    find_peers_by_name,
    list_peers,
    peer_by_account,
)


# ---------------------------------------------------------------------------
# PeerSpec validation
# ---------------------------------------------------------------------------


class TestPeerSpecValidation:
    """Peer ids are filesystem-safe and follow a strict grammar."""

    def test_valid_simple_id(self) -> None:
        spec = PeerSpec(id="ada-lovelace", name="Ada Lovelace")
        assert spec.id == "ada-lovelace"
        assert spec.kind is PeerKind.HUMAN

    def test_valid_programmatic_id(self) -> None:
        spec = PeerSpec(id="peer-7f3a", name="")
        assert spec.id == "peer-7f3a"

    def test_id_with_underscore_and_dot(self) -> None:
        PeerSpec(id="alice_jones.v2", name="Alice")

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",  # empty
            ".hidden",  # leading dot
            "-leading-dash",  # leading dash
            "_underscore",  # leading underscore
            "has space",  # whitespace
            "has/slash",  # path separator
            "has*glob",  # shell-fragile
        ],
    )
    def test_invalid_ids_rejected(self, bad_id: str) -> None:
        with pytest.raises(Exception):
            PeerSpec(id=bad_id, name="x")

    def test_default_kind_is_human(self) -> None:
        spec = PeerSpec(id="x", name="")
        assert spec.kind is PeerKind.HUMAN

    def test_empty_name_allowed(self) -> None:
        spec = PeerSpec(id="dependabot", name="", kind=PeerKind.BOT)
        assert spec.name == ""
        assert spec.kind is PeerKind.BOT


# ---------------------------------------------------------------------------
# PeerRegistry
# ---------------------------------------------------------------------------


def _peer(
    pid: str,
    *,
    name: str = "",
    kind: PeerKind = PeerKind.HUMAN,
    accounts: list[tuple[str, str]] | None = None,
) -> PeerSpec:
    """Compact factory for ``PeerSpec`` instances in tests."""
    return PeerSpec(
        id=pid,
        name=name,
        kind=kind,
        accounts=[
            PeerAccount(service=svc, account_id=aid)
            for svc, aid in (accounts or [])
        ],
    )


class TestPeerRegistry:
    def test_lookup_by_immutable_id(self) -> None:
        reg = PeerRegistry(
            [_peer("alice", accounts=[("github", "12345")])],
        )
        actor = ActorIdentity(
            service="github",
            account_id="12345",
            secondary_account_ids=("alice-handle",),
        )
        match = reg.lookup_actor(actor)
        assert match is not None
        assert match.id == "alice"

    def test_lookup_by_secondary_handle(self) -> None:
        # Operator wrote the textual handle; events carry immutable id
        # in account_id and login as a secondary.  Should still match.
        reg = PeerRegistry(
            [_peer("alice", accounts=[("github", "alice-handle")])],
        )
        actor = ActorIdentity(
            service="github",
            account_id="12345",
            secondary_account_ids=("alice-handle",),
        )
        match = reg.lookup_actor(actor)
        assert match is not None
        assert match.id == "alice"

    def test_lookup_miss_when_account_unknown(self) -> None:
        reg = PeerRegistry(
            [_peer("alice", accounts=[("github", "12345")])],
        )
        actor = ActorIdentity(
            service="github",
            account_id="99999",
            secondary_account_ids=("stranger",),
        )
        assert reg.lookup_actor(actor) is None

    def test_service_namespace_is_per_service(self) -> None:
        """An account_id collision across services must not cross-match."""
        reg = PeerRegistry(
            [
                _peer("alice", accounts=[("github", "12345")]),
                _peer("bob", accounts=[("gitlab", "12345")]),
            ],
        )
        gh_actor = ActorIdentity(service="github", account_id="12345")
        gl_actor = ActorIdentity(service="gitlab", account_id="12345")
        gh_match = reg.lookup_actor(gh_actor)
        gl_match = reg.lookup_actor(gl_actor)
        assert gh_match is not None and gh_match.id == "alice"
        assert gl_match is not None and gl_match.id == "bob"

    def test_duplicate_peer_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate peer id"):
            PeerRegistry([_peer("alice"), _peer("alice")])

    def test_account_owned_by_two_peers_rejected(self) -> None:
        with pytest.raises(ValueError, match="claimed by both peer"):
            PeerRegistry(
                [
                    _peer("alice", accounts=[("github", "12345")]),
                    _peer("bob", accounts=[("github", "12345")]),
                ],
            )

    def test_lookup_account_handles_both_forms(self) -> None:
        reg = PeerRegistry(
            [_peer("alice", accounts=[("github", "alice-handle")])],
        )
        assert reg.lookup_account("github", "alice-handle") is not None
        assert reg.lookup_account("github", "missing") is None
        assert reg.lookup_account("nosuch-service", "anything") is None

    def test_find_by_name_substring_case_insensitive(self) -> None:
        reg = PeerRegistry(
            [
                _peer("a", name="Tess McGee"),
                _peer("b", name="Bob Roberts"),
                _peer("c", name="tessellation Bot"),
            ],
        )
        results = sorted(p.id for p in reg.find_by_name("tess"))
        assert results == ["a", "c"]

    def test_find_by_name_empty_query_returns_empty(self) -> None:
        reg = PeerRegistry([_peer("a", name="Anything")])
        assert reg.find_by_name("") == []

    def test_all_peers_sorted_by_id(self) -> None:
        reg = PeerRegistry([_peer("zeta"), _peer("alpha"), _peer("mu")])
        assert [p.id for p in reg.all_peers()] == ["alpha", "mu", "zeta"]

    def test_get_returns_spec_or_none(self) -> None:
        reg = PeerRegistry([_peer("alice")])
        assert reg.get("alice").id == "alice"  # type: ignore[union-attr]
        assert reg.get("missing") is None


# ---------------------------------------------------------------------------
# wrap_external
# ---------------------------------------------------------------------------


class TestWrapExternal:
    def test_envelope_shape_with_actor(self) -> None:
        actor = ActorIdentity(
            service="github",
            account_id="12345",
            secondary_account_ids=("alice",),
        )
        rendered = wrap_external(
            body="Please fix the typo.",
            actor=actor,
            source="github",
            kind="comment",
            peer_status=PeerStatus.PEER,
            timestamp="2026-04-30T12:34Z",
            nonce="deadbeef",
        )
        # Opening / closing markers carry the same nonce.
        assert "[external-content nonce=deadbeef" in rendered
        assert rendered.endswith("[/external-content nonce=deadbeef]")
        # Marker carries metadata for the agent.
        assert "source=github" in rendered
        assert "actor=@alice" in rendered
        assert "peer=yes" in rendered
        assert "kind=comment" in rendered
        # Body lines are blockquoted.
        assert "> @alice (2026-04-30T12:34Z):" in rendered
        assert "> Please fix the typo." in rendered

    def test_envelope_with_no_actor(self) -> None:
        rendered = wrap_external(
            body="System ping.",
            actor=None,
            source="harness",
            kind="harness_note",
            peer_status=PeerStatus.UNKNOWN,
            nonce="cafef00d",
        )
        assert "actor=(unknown actor)" in rendered
        assert "peer=unknown" in rendered
        assert "> (unknown actor):" in rendered

    def test_empty_body_renders_no_body_line(self) -> None:
        rendered = wrap_external(
            body="",
            actor=None,
            source="x",
            kind="comment",
            nonce="0",
        )
        assert "> (no body)" in rendered

    def test_blank_body_lines_stay_in_blockquote(self) -> None:
        rendered = wrap_external(
            body="line one\n\nline two",
            actor=None,
            source="x",
            kind="comment",
            nonce="0",
        )
        # Blank lines inside the body get a bare ``>`` so the
        # blockquote does not break.
        assert "\n>\n> line two" in rendered

    def test_distinct_calls_get_distinct_nonces(self) -> None:
        a = wrap_external(body="x", actor=None, source="s", kind="k")
        b = wrap_external(body="x", actor=None, source="s", kind="k")
        # Two random 32-bit nonces colliding is astronomical; if
        # this ever flakes we have a deeper problem.
        assert a != b

    def test_peer_status_unknown_distinguishable_from_no(self) -> None:
        no = wrap_external(
            body="x", actor=None, source="s", kind="k",
            peer_status=PeerStatus.NON_PEER, nonce="1",
        )
        unk = wrap_external(
            body="x", actor=None, source="s", kind="k",
            peer_status=PeerStatus.UNKNOWN, nonce="1",
        )
        assert "peer=no" in no
        assert "peer=unknown" in unk
        assert no != unk


# ---------------------------------------------------------------------------
# describe_actor_for_log: privacy guard
# ---------------------------------------------------------------------------


class TestDescribeActorForLog:
    def test_matched_peer_uses_peer_id(self) -> None:
        actor = ActorIdentity(
            service="github",
            account_id="12345",
            display_name="Display Name That Should Not Leak",
        )
        out = describe_actor_for_log(actor, peer_id="alice")
        assert out == "<peer:alice>"
        assert "Display Name" not in out

    def test_unmatched_actor_uses_immutable_id(self) -> None:
        actor = ActorIdentity(
            service="github",
            account_id="12345",
            secondary_account_ids=("login-handle",),
            display_name="A Person",
        )
        out = describe_actor_for_log(actor)
        # No display name leaked into log output, no @login form
        # either -- log format is namespace:id only.
        assert out == "<actor:github:12345>"

    def test_none_actor(self) -> None:
        assert describe_actor_for_log(None) == "<actor:unknown>"


# ---------------------------------------------------------------------------
# TriggerAuthorizationPolicy
# ---------------------------------------------------------------------------


def _raw(
    *,
    source: str = "github",
    kind: EventKind = EventKind.CONVERSATIONAL,
    actor: ActorIdentity | None = None,
    items: tuple[ContextItem, ...] = (),
    summary: str = "Something happened",
) -> RawIncomingEvent:
    return RawIncomingEvent(
        source=source,
        session_key=SessionKey("k"),
        kind=kind,
        primary_actor=actor,
        summary=summary,
        items=items,
    )


class TestTriggerAuthorizationPolicy:
    def test_system_events_always_delivered(self) -> None:
        policy = TriggerAuthorizationPolicy(PeerRegistry([]))
        decision = policy.decide(_raw(kind=EventKind.SYSTEM, actor=None))
        assert isinstance(decision, Deliver)
        assert decision.peer is None

    def test_conversational_from_peer_delivered(self) -> None:
        peer = _peer("alice", accounts=[("github", "12345")])
        policy = TriggerAuthorizationPolicy(PeerRegistry([peer]))
        actor = ActorIdentity(service="github", account_id="12345")
        decision = policy.decide(_raw(actor=actor))
        assert isinstance(decision, Deliver)
        assert decision.peer is not None
        assert decision.peer.id == "alice"

    def test_conversational_from_non_peer_dropped(self) -> None:
        policy = TriggerAuthorizationPolicy(PeerRegistry([]))
        actor = ActorIdentity(service="github", account_id="99999")
        decision = policy.decide(_raw(actor=actor))
        assert isinstance(decision, Drop)
        assert "non-peer" in decision.reason

    def test_conversational_with_no_actor_dropped(self) -> None:
        policy = TriggerAuthorizationPolicy(PeerRegistry([]))
        decision = policy.decide(_raw(actor=None))
        assert isinstance(decision, Drop)

    def test_structural_from_peer_delivered_no_banner(self) -> None:
        peer = _peer("alice", accounts=[("github", "12345")])
        policy = TriggerAuthorizationPolicy(PeerRegistry([peer]))
        actor = ActorIdentity(service="github", account_id="12345")
        decision = policy.decide(_raw(kind=EventKind.STRUCTURAL, actor=actor))
        assert isinstance(decision, Deliver)
        assert decision.peer is not None

    def test_structural_from_non_peer_delivers_with_banner_by_default(
        self,
    ) -> None:
        policy = TriggerAuthorizationPolicy(PeerRegistry([]))
        actor = ActorIdentity(
            service="github",
            account_id="99999",
            secondary_account_ids=("stranger",),
        )
        decision = policy.decide(_raw(kind=EventKind.STRUCTURAL, actor=actor))
        assert isinstance(decision, DeliverWithBanner)
        # Banner names the actor and warns the agent to not act on
        # body instructions absent peer authorization.
        assert "@stranger" in decision.banner
        assert "non-peer" in decision.banner

    def test_structural_from_non_peer_dropped_when_carveout_disabled(
        self,
    ) -> None:
        policy = TriggerAuthorizationPolicy(
            PeerRegistry([]),
            source_policies={
                "github": SourceTriggerPolicy(
                    deliver_structural_from_non_peers=False,
                ),
            },
        )
        actor = ActorIdentity(service="github", account_id="99999")
        decision = policy.decide(_raw(kind=EventKind.STRUCTURAL, actor=actor))
        assert isinstance(decision, Drop)
        assert "delivery disabled" in decision.reason

    def test_unregistered_bot_dropped_even_on_structural_event(self) -> None:
        policy = TriggerAuthorizationPolicy(PeerRegistry([]))
        actor = ActorIdentity(
            service="github",
            account_id="dependabot[bot]",
            is_bot=True,
        )
        decision = policy.decide(_raw(kind=EventKind.STRUCTURAL, actor=actor))
        assert isinstance(decision, Drop)
        assert "unregistered bot" in decision.reason

    def test_bot_registered_as_bot_peer_delivered(self) -> None:
        peer = _peer(
            "dependabot",
            kind=PeerKind.BOT,
            accounts=[("github", "49699333")],
        )
        policy = TriggerAuthorizationPolicy(PeerRegistry([peer]))
        actor = ActorIdentity(
            service="github",
            account_id="49699333",
            is_bot=True,
        )
        decision = policy.decide(_raw(actor=actor))
        assert isinstance(decision, Deliver)
        assert decision.peer is not None
        assert decision.peer.kind is PeerKind.BOT

    def test_bot_matching_a_human_peer_still_dropped(self) -> None:
        """Mismatch in ``kind`` blocks the bot even if ids line up.

        Operator declared the account as a human peer; an event flagged
        ``is_bot=True`` for the same account is treated as suspicious
        and dropped.  This is the confused-deputy guard.
        """
        peer = _peer(
            "alice",
            kind=PeerKind.HUMAN,
            accounts=[("github", "12345")],
        )
        policy = TriggerAuthorizationPolicy(PeerRegistry([peer]))
        actor = ActorIdentity(
            service="github",
            account_id="12345",
            is_bot=True,
        )
        decision = policy.decide(_raw(actor=actor))
        assert isinstance(decision, Drop)


# ---------------------------------------------------------------------------
# NotificationFormatter
# ---------------------------------------------------------------------------


class TestNotificationFormatter:
    def _formatter(
        self, *peers: PeerSpec, source_policies: dict | None = None,
    ) -> NotificationFormatter:
        registry = PeerRegistry(list(peers))
        policy = TriggerAuthorizationPolicy(
            registry, source_policies=source_policies,
        )
        return NotificationFormatter(peer_registry=registry, policy=policy)

    def test_drop_returns_formatter_drop(self) -> None:
        fmt = self._formatter()
        actor = ActorIdentity(service="github", account_id="99999")
        result = fmt.process(_raw(actor=actor))
        assert isinstance(result, FormatterDrop)
        assert "non-peer" in result.reason

    def test_deliver_renders_summary_and_envelope(self) -> None:
        peer = _peer("alice", name="Alice", accounts=[("github", "12345")])
        fmt = self._formatter(peer)
        actor = ActorIdentity(
            service="github",
            account_id="12345",
            secondary_account_ids=("alice-handle",),
        )
        item = ContextItem(
            body="Please review.",
            kind=ContextItemKind.COMMENT,
            actor=actor,
        )
        result = fmt.process(
            _raw(actor=actor, items=(item,), summary="Comment on issue #1"),
        )
        assert isinstance(result, FormatterDelivery)
        content = result.event.content
        assert "Comment on issue #1" in content
        assert "[external-content" in content
        assert "peer=yes" in content
        assert "> Please review." in content

    def test_deliver_with_banner_inserts_warning_before_envelope(self) -> None:
        fmt = self._formatter()
        actor = ActorIdentity(
            service="github",
            account_id="99999",
            secondary_account_ids=("stranger",),
        )
        item = ContextItem(
            body="Look at this!",
            kind=ContextItemKind.ISSUE_BODY,
            actor=actor,
        )
        result = fmt.process(
            _raw(
                kind=EventKind.STRUCTURAL,
                actor=actor,
                items=(item,),
                summary="Issue #5 opened",
            ),
        )
        assert isinstance(result, FormatterDelivery)
        content = result.event.content
        # Banner appears before envelope; envelope is labelled non-peer.
        banner_idx = content.find("non-peer")
        envelope_idx = content.find("[external-content")
        assert banner_idx != -1
        assert envelope_idx != -1
        assert banner_idx < envelope_idx
        assert "peer=no" in content

    def test_per_item_peer_status_resolved_independently(self) -> None:
        """A thread can mix peer and non-peer authors; each chunk gets its own label."""
        peer = _peer("alice", accounts=[("github", "12345")])
        fmt = self._formatter(peer)
        peer_actor = ActorIdentity(service="github", account_id="12345")
        stranger = ActorIdentity(service="github", account_id="99999")
        items = (
            ContextItem(
                body="From a peer.",
                kind=ContextItemKind.COMMENT,
                actor=peer_actor,
            ),
            ContextItem(
                body="From a stranger.",
                kind=ContextItemKind.COMMENT,
                actor=stranger,
            ),
        )
        # Primary actor is the peer, so the trigger decision is
        # ``Deliver`` (not banner).  Per-item peer status is still
        # resolved independently for the second item.
        result = fmt.process(_raw(actor=peer_actor, items=items))
        assert isinstance(result, FormatterDelivery)
        content = result.event.content
        assert content.count("[external-content") == 2
        assert "peer=yes" in content
        assert "peer=no" in content


# ---------------------------------------------------------------------------
# thorn.tools.peers
# ---------------------------------------------------------------------------


@pytest.fixture
def peer_runtime(tmp_path: Path) -> Runtime:
    """A runtime pre-populated with a small peer registry."""
    runtime = Runtime(
        provider=MockProvider(),
        workspace_root=tmp_path / "ws",
    )
    runtime.peer_registry = PeerRegistry(
        [
            _peer(
                "alice",
                name="Alice Anders",
                accounts=[("github", "12345"), ("gitlab", "alice-gl")],
            ),
            _peer("bob", name="Bob Builder", accounts=[("github", "67890")]),
            _peer("dependabot", name="", kind=PeerKind.BOT,
                  accounts=[("github", "49699333")]),
        ],
    )
    return runtime


@pytest.fixture
def peer_ctx(peer_runtime: Runtime) -> Iterator[ExecutionContext]:
    base = peer_runtime.create_context()
    scoped = base.push_scope(
        "peer-tools-test",
        agent=Agent(id=AgentID("test-agent"), name="tester"),
    )
    token = set_context(scoped)
    try:
        yield scoped
    finally:
        reset_context(token)


class TestPeerByAccountTool:
    async def test_immutable_id_match(self, peer_ctx: ExecutionContext) -> None:
        result = await peer_by_account("github", "12345")
        assert isinstance(result, Peer)
        assert result.id == "alice"
        assert result.name == "Alice Anders"
        assert result.kind is PeerKind.HUMAN

    async def test_textual_handle_match(self, peer_ctx: ExecutionContext) -> None:
        result = await peer_by_account("gitlab", "alice-gl")
        assert result is not None
        assert result.id == "alice"

    async def test_miss_returns_none(self, peer_ctx: ExecutionContext) -> None:
        assert await peer_by_account("github", "doesnt-exist") is None

    async def test_unknown_service_returns_none(
        self, peer_ctx: ExecutionContext,
    ) -> None:
        assert await peer_by_account("nosuch-service", "12345") is None

    async def test_returns_no_credentials(
        self, peer_ctx: ExecutionContext,
    ) -> None:
        result = await peer_by_account("github", "12345")
        assert result is not None
        # Public Peer shape has only id/name/kind/accounts.
        assert set(result.model_dump().keys()) == {
            "id", "name", "kind", "accounts",
        }
        for account in result.accounts:
            assert set(account.model_dump().keys()) == {"service", "account_id"}


class TestFindPeersByNameTool:
    async def test_substring_match(self, peer_ctx: ExecutionContext) -> None:
        results = await find_peers_by_name("alice")
        assert len(results) == 1
        assert results[0].id == "alice"

    async def test_case_insensitive(self, peer_ctx: ExecutionContext) -> None:
        results = await find_peers_by_name("ANDERS")
        assert {p.id for p in results} == {"alice"}

    async def test_no_match_returns_empty(
        self, peer_ctx: ExecutionContext,
    ) -> None:
        assert await find_peers_by_name("nobody") == []

    async def test_empty_query_returns_empty(
        self, peer_ctx: ExecutionContext,
    ) -> None:
        # An empty query against the empty string would otherwise match
        # every peer; the registry rejects this case explicitly.
        assert await find_peers_by_name("") == []


class TestListPeersTool:
    async def test_returns_all_peers_sorted_by_id(
        self, peer_ctx: ExecutionContext,
    ) -> None:
        results = await list_peers()
        ids = [p.id for p in results]
        assert ids == sorted(ids)
        assert {"alice", "bob", "dependabot"} <= set(ids)

    async def test_bot_peer_kind_preserved(
        self, peer_ctx: ExecutionContext,
    ) -> None:
        results = await list_peers()
        bot = next(p for p in results if p.id == "dependabot")
        assert bot.kind is PeerKind.BOT


class TestPeerCrossConfigValidation:
    """Cross-config peer/service validation runs against the resolved set.

    The pydantic validator on :class:`GatewayConfig` deliberately does
    *not* check ``peer.account.service`` against ``self.forges`` --
    that list is incomplete at parse time, before
    :func:`_resolve_forges_and_projects` synthesises forges for
    project fork URLs.  Validation moved into the resolver so a peer
    that names a synthesised forge is accepted, while a peer that
    names something nothing in the config produces is still rejected
    with a clear error.
    """

    def test_peer_referencing_synthesised_forge_passes(self) -> None:
        """No explicit ``forges:`` block; the github forge is synthesised."""
        from thorn.gateway._config import (
            ForkSpec,
            GatewayConfig,
            ProjectSpec,
            _resolve_forges_and_projects,
        )

        cfg = GatewayConfig(
            projects=[
                ProjectSpec(
                    name="tt",
                    forks=[ForkSpec(url="https://github.com/x/y")],
                ),
            ],
            peers=[
                PeerSpec(
                    id="alice",
                    accounts=[PeerAccount(service="github", account_id="alice")],
                ),
            ],
        )

        # Should not raise, and the synthesised forge should be in the
        # resolved list.
        forge_specs, _ = _resolve_forges_and_projects(cfg)
        assert {f.name for f in forge_specs} == {"github"}

    def test_peer_referencing_explicit_forge_passes(self) -> None:
        from thorn.gateway._config import (
            ForgeSpec,
            GatewayConfig,
            _resolve_forges_and_projects,
        )

        cfg = GatewayConfig(
            forges=[ForgeSpec(name="my-gh", url="https://github.com")],
            peers=[
                PeerSpec(
                    id="alice",
                    accounts=[PeerAccount(service="my-gh", account_id="alice")],
                ),
            ],
        )

        # No exception; resolver returns the explicit forge unchanged.
        forge_specs, _ = _resolve_forges_and_projects(cfg)
        assert "my-gh" in {f.name for f in forge_specs}

    def test_peer_referencing_unknown_service_rejected(self) -> None:
        from thorn.gateway._config import (
            ForkSpec,
            GatewayConfig,
            ProjectSpec,
            _resolve_forges_and_projects,
        )

        cfg = GatewayConfig(
            projects=[
                ProjectSpec(
                    name="tt",
                    forks=[ForkSpec(url="https://github.com/x/y")],
                ),
            ],
            peers=[
                PeerSpec(
                    id="alice",
                    accounts=[
                        PeerAccount(service="githbu", account_id="alice"),
                    ],
                ),
            ],
        )

        with pytest.raises(ValueError) as excinfo:
            _resolve_forges_and_projects(cfg)

        msg = str(excinfo.value)
        # Error names the offending peer, the bad service, and the set
        # of services the operator could have meant.
        assert "alice" in msg
        assert "githbu" in msg
        assert "github" in msg  # in the "Known services" list

    def test_pydantic_validation_no_longer_inspects_peers(self) -> None:
        """Parsing a config with peers but no matching forge succeeds.

        The cross-config check moved out of the pydantic validator
        and into the resolver.  Loading the model alone (e.g. for a
        partial inspection or for tooling that does not run resolution)
        should therefore succeed even when the operator-declared
        ``forges`` list does not yet contain the relevant entry.
        """
        from thorn.gateway._config import (
            ForkSpec,
            GatewayConfig,
            ProjectSpec,
        )

        cfg = GatewayConfig(
            projects=[
                ProjectSpec(
                    name="tt",
                    forks=[ForkSpec(url="https://github.com/x/y")],
                ),
            ],
            peers=[
                PeerSpec(
                    id="alice",
                    accounts=[PeerAccount(service="github", account_id="alice")],
                ),
            ],
        )

        # Parse-time forges is the operator-declared list (empty),
        # but the model loaded fine because cross-config validation
        # was deferred to resolution time.
        assert cfg.forges == []
        assert cfg.peers[0].id == "alice"

    def test_validate_peers_against_services_directly(self) -> None:
        """The exposed helper takes a service-name set and validates."""
        from thorn.gateway._config import validate_peers_against_services

        peers = [
            PeerSpec(
                id="alice",
                accounts=[PeerAccount(service="gh", account_id="alice")],
            ),
        ]
        # Hit: should not raise.
        validate_peers_against_services(peers, {"gh", "gl"})
        # Miss: raises with the bad service named.
        with pytest.raises(ValueError, match="bogus"):
            validate_peers_against_services(peers, {"bogus"})

    def test_peer_with_no_accounts_warns_but_passes(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A peer with zero accounts is allowed (mid-edit) but warned about."""
        from thorn.gateway._config import validate_peers_against_services

        peers = [PeerSpec(id="alice", accounts=[])]
        with caplog.at_level("WARNING", logger="thorn.gateway._config"):
            validate_peers_against_services(peers, set())
        joined = "\n".join(r.message for r in caplog.records)
        assert "alice" in joined
        assert "no accounts" in joined.lower()


class TestGatewaySourcePoliciesUseResolvedForges:
    """``Gateway.__init__`` keys per-source policy off the resolved forge list.

    The same layering bug that hit peer cross-config validation also
    affected the per-source policy loop (``deliver_structural_from_non_peers``
    and any future per-forge knobs).  This test pins down that
    Gateway now walks the resolved forge list -- including forges
    synthesised from project fork URLs -- when building its trigger
    policy's per-source dict.
    """

    def test_synthesised_forge_appears_in_source_policies(
        self, tmp_path: Path,
    ) -> None:
        from thorn.core._provider import MockProvider
        from thorn.gateway._config import (
            ForkSpec,
            GatewayConfig,
            ProjectSpec,
        )
        from thorn.gateway._gateway import Gateway

        cfg = GatewayConfig(
            projects=[
                ProjectSpec(
                    name="tt",
                    forks=[ForkSpec(url="https://github.com/x/y")],
                ),
            ],
        )

        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path / "ws",
        )
        gateway = Gateway(runtime=runtime, sources=[], gateway_config=cfg)

        # ``deliver_structural_from_non_peers=True`` is the default,
        # but the important pin is that a per-source policy entry
        # *exists* for the synthesised forge -- so a future per-forge
        # knob set on a fork-derived forge will be honoured rather
        # than silently falling back to the default.
        source_policies = gateway._trigger_policy._source_policies  # noqa: SLF001
        assert "github" in source_policies

    def test_explicit_per_forge_knob_propagates_to_resolved_walk(
        self, tmp_path: Path,
    ) -> None:
        """An operator-set knob on the explicit forge is honoured."""
        from thorn.core._provider import MockProvider
        from thorn.gateway._config import (
            ForgeSpec,
            ForkSpec,
            GatewayConfig,
            ProjectSpec,
        )
        from thorn.gateway._gateway import Gateway

        cfg = GatewayConfig(
            forges=[
                ForgeSpec(
                    url="https://github.com",
                    deliver_structural_from_non_peers=False,
                ),
            ],
            projects=[
                ProjectSpec(
                    name="tt",
                    forks=[ForkSpec(url="https://github.com/x/y")],
                ),
            ],
        )

        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path / "ws",
        )
        gateway = Gateway(runtime=runtime, sources=[], gateway_config=cfg)

        policy = gateway._trigger_policy._source_policies["github"]  # noqa: SLF001
        assert policy.deliver_structural_from_non_peers is False


class TestPeerToolsWithoutRuntime:
    """Tools must fail clearly when no Runtime is in scope."""

    async def test_no_runtime_raises(self) -> None:
        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            with pytest.raises(RuntimeError, match="Runtime"):
                await peer_by_account("github", "12345")
        finally:
            reset_context(token)


class TestGatewayAgentSystemPrompt:
    """The gateway-wide trust-model prompt is collected via MRO.

    The universal guidance must reach every concrete gateway-resident
    agent without each subclass having to re-declare it; this verifies
    that the MRO walk picks it up from :class:`GatewayAgent` and
    prepends it to the role-specific prompt of subclasses like
    :class:`ProjectCoordinator`.
    """

    def test_project_coordinator_inherits_universal_prompt(self) -> None:
        from thorn.gateway._agents import (
            _COORDINATOR_SYSTEM_PROMPT,
            _GATEWAY_AGENT_UNIVERSAL_PROMPT,
            ProjectCoordinator,
        )

        prompts = ProjectCoordinator._collect_system_prompts()
        assert _GATEWAY_AGENT_UNIVERSAL_PROMPT in prompts
        assert _COORDINATOR_SYSTEM_PROMPT in prompts
        # MRO is outermost-first, so the base-class universal prompt
        # appears before the subclass role prompt.
        assert prompts.index(_GATEWAY_AGENT_UNIVERSAL_PROMPT) < prompts.index(
            _COORDINATOR_SYSTEM_PROMPT,
        )

    def test_universal_prompt_covers_required_topics(self) -> None:
        """The prompt must mention every topic the threat model relies on."""
        from thorn.gateway._agents import _GATEWAY_AGENT_UNIVERSAL_PROMPT

        prompt = _GATEWAY_AGENT_UNIVERSAL_PROMPT
        # Envelope shape callout.
        assert "[external-content" in prompt
        assert "peer=" in prompt
        # Three peer-status branches.
        assert "peer=yes" in prompt
        assert "peer=no" in prompt
        assert "peer=unknown" in prompt
        # Authority claim guard.
        assert "authority" in prompt.lower()
        # Bot confused-deputy guard.
        assert "bot" in prompt.lower()
        # Peer-notes location and no-secrets rule.
        assert "~/peers/" in prompt
        # Self-disclosure boundary.
        assert "Self-disclosure" in prompt or "self-disclosure" in prompt

    def test_gateway_agent_has_peer_tools_by_default(self) -> None:
        """Peer-lookup tools are part of the universal toolkit, not opt-in."""
        from thorn.gateway._agents import GatewayAgent, ProjectCoordinator

        gateway_tool_names = {
            getattr(t, "__name__", str(t))
            for t in GatewayAgent._collect_tools()
        }
        coordinator_tool_names = {
            getattr(t, "__name__", str(t))
            for t in ProjectCoordinator._collect_tools()
        }
        for required in ("peer_by_account", "find_peers_by_name", "list_peers"):
            assert required in gateway_tool_names
            assert required in coordinator_tool_names


class TestPeerToolsWithEmptyRegistry:
    """An empty registry is the strict default and yields zero matches."""

    async def test_empty_registry_returns_none_and_empty_lists(
        self, tmp_path: Path,
    ) -> None:
        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path / "ws",
        )
        # Default Runtime constructs an empty PeerRegistry.
        ctx = runtime.create_context().push_scope(
            "test",
            agent=Agent(id=AgentID("a"), name="a"),
        )
        token = set_context(ctx)
        try:
            assert await peer_by_account("github", "12345") is None
            assert await find_peers_by_name("anyone") == []
            assert await list_peers() == []
        finally:
            reset_context(token)
