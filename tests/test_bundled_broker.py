"""Unit tests for :mod:`thorn.gateway._bundled_broker`.

These tests exercise the supervisor's bring-up / shutdown flow in
isolation: the OCI-runtime detection, compose subprocess invocations,
and OneCLI HTTP calls are all replaced with record-and-replay fakes
so the suite runs in milliseconds and never touches docker / podman.

The opt-in real-runtime smoke test
(:mod:`tests.sandbox.test_smoke_bundled_broker`) covers the
docker-actually-runs-the-stack side of the contract.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from thorn.gateway._bundled_broker import (
    BundledBrokerError,
    BundledBrokerSupervisor,
    _generate_compose_project_name,
    _parse_api_key_response,
    _split_compose_port_output,
)
from thorn.gateway._resources_helper import read_bundled_broker_compose_text

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _ComposeRecorder:
    """Record-and-replay fake for the supervisor's subprocess runner.

    Captures every (argv, env, timeout) invocation in order, and
    returns a scripted ``(rc, stdout, stderr)`` triple.  The script
    is keyed on the trailing compose verb so individual tests can
    declare "for ``up``, return X; for ``port``, return Y; for
    ``down``, return Z" without caring about the unchanging argv
    prefix the supervisor synthesises.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, str], float]] = []
        # verb -> list of scripted responses (FIFO).  Lets one verb
        # be exercised multiple times in a single test (e.g. two
        # ``port`` calls for admin + proxy).
        self._responses: dict[str, list[tuple[int, str, str]]] = {}

    def script(
        self,
        verb: str,
        *,
        rc: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self._responses.setdefault(verb, []).append((rc, stdout, stderr))

    async def __call__(
        self,
        argv: tuple[str, ...],
        env: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, str, str]:
        self.calls.append((argv, env, timeout_s))
        # The supervisor's argv always ends with the verb tuple it
        # passed to ``_run_compose_capturing``.  The verb of interest
        # for scripting is the first non-flag token after the cached
        # argv prefix; the supervisor uses ``up``, ``port``, ``down``
        # as the leading verbs.
        verb = self._verb_of(argv)
        responses = self._responses.get(verb)
        if not responses:
            raise AssertionError(
                f"unexpected compose verb {verb!r} (argv={argv!r}); "
                f"scripted={list(self._responses)}",
            )
        return responses.pop(0)

    @staticmethod
    def _verb_of(argv: tuple[str, ...]) -> str:
        # argv layout: (binary, "compose", "-p", project, "-f", path, verb, ...)
        # We want index 6 (the verb).  Defensive against shape drift.
        for i, token in enumerate(argv):
            if token == "-f" and i + 2 < len(argv):
                return argv[i + 2]
        return argv[-1]


class TestBundledComposeResource:
    def test_onecli_can_use_host_gateway_and_host_ca_bundle(self) -> None:
        compose = read_bundled_broker_compose_text()
        assert "ONECLI_HOST_GATEWAY_HOST" in compose
        assert ":host-gateway" in compose
        assert "ONECLI_HOST_CA_BUNDLE" in compose
        assert "SSL_CERT_FILE: /etc/ssl/certs/ca-certificates.crt" in compose
        assert (
            "NODE_EXTRA_CA_CERTS: /etc/ssl/certs/ca-certificates.crt"
            in compose
        )


def _http_factory(handler: httpx.MockTransport) -> Any:
    """Build a factory that produces an ``httpx.Client`` bound to *handler*.

    Per-call client is fine here: the supervisor only makes a small
    constant number of HTTP requests during bring-up and the cost of
    spinning a client up around a mock transport is negligible.
    """

    def _factory() -> httpx.Client:
        return httpx.Client(transport=handler, timeout=2.0)

    return _factory


def _runtime_factory(name: str = "podman") -> Any:
    """Compose-runtime factory that returns a deterministic name + path.

    Avoids touching ``shutil.which`` / the real PATH so tests are
    fully hermetic.
    """
    fake_path = f"/usr/bin/{name}"

    def _factory() -> tuple[str, tuple[str, ...]]:
        return name, (fake_path, "compose")

    return _factory


def _ok_health_handler(api_key_responses: list[tuple[int, dict[str, Any] | str]]) -> httpx.MockTransport:
    """Mock transport that serves /api/health 200 and a scripted key endpoint.

    *api_key_responses* is a FIFO list of ``(status_code, body)`` for
    each call to ``/api/user/api-key`` or ``/api/user/api-key/regenerate``,
    in call order.  Body may be a dict (JSON-serialised) or a string
    (returned verbatim).
    """
    pending = list(api_key_responses)

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path in (
            "/api/user/api-key",
            "/api/user/api-key/regenerate",
        ):
            if not pending:
                raise AssertionError(
                    f"unexpected extra api-key call to {request.url.path}",
                )
            status, body = pending.pop(0)
            if isinstance(body, dict):
                return httpx.Response(status, json=body)
            return httpx.Response(status, text=body)
        raise AssertionError(f"unexpected URL: {request.url}")

    return httpx.MockTransport(_handler)


def _make_supervisor(
    *,
    recorder: _ComposeRecorder,
    handler: httpx.MockTransport,
    runtime: str = "podman",
) -> BundledBrokerSupervisor:
    return BundledBrokerSupervisor(
        bind_host="127.0.0.1",
        # Deliberately small timeouts: every test scripts the
        # responses, so the only way these would fire is if the
        # supervisor mis-routed a call.  Keep the suite fast.
        health_timeout_s=2.0,
        health_poll_interval_s=0.01,
        compose_runtime_factory=_runtime_factory(runtime),
        http_client_factory=_http_factory(handler),
        subprocess_runner=recorder,
    )


# ---------------------------------------------------------------------------
# _generate_compose_project_name
# ---------------------------------------------------------------------------


class TestGenerateComposeProjectName:
    """The bundled-prefix invariant is what makes orphan cleanup work.

    ``thorn broker status`` / ``thorn broker down`` filter compose
    projects by this prefix, so any drift here would silently break
    the cleanup commands.
    """

    def test_name_uses_bundled_prefix(self) -> None:
        name = _generate_compose_project_name()
        assert name.startswith("thorn-broker-")

    def test_names_are_unique_across_calls(self) -> None:
        # Per-process uniqueness is what lets concurrent ``thorn serve``
        # runs share a host without colliding on compose project name.
        names = {_generate_compose_project_name() for _ in range(50)}
        assert len(names) == 50


# ---------------------------------------------------------------------------
# _split_compose_port_output
# ---------------------------------------------------------------------------


class TestSplitComposePortOutput:
    """Compose's ``port`` verb is the supervisor's port-discovery seam.

    Both docker and podman have shipped subtly different output shapes
    over the years (trailing newlines, leading warnings on some
    podman versions, IPv6 brackets); the parser needs to be lenient.
    """

    def test_simple_ipv4(self) -> None:
        assert _split_compose_port_output("127.0.0.1:54321\n") == ("127.0.0.1", 54321)

    def test_ipv6(self) -> None:
        assert _split_compose_port_output("[::1]:54321") == ("::1", 54321)

    def test_skips_whitespace_lines(self) -> None:
        # Some podman versions emit an empty leading line then the port.
        assert _split_compose_port_output("\n127.0.0.1:8080\n") == ("127.0.0.1", 8080)

    def test_empty_output_raises(self) -> None:
        with pytest.raises(BundledBrokerError, match="empty output"):
            _split_compose_port_output("")

    def test_garbage_raises(self) -> None:
        with pytest.raises(BundledBrokerError):
            _split_compose_port_output("not-a-port-line")


# ---------------------------------------------------------------------------
# _parse_api_key_response
# ---------------------------------------------------------------------------


class TestParseApiKeyResponse:
    """OneCLI's admin-key endpoints are the single point of trust for
    the bundled-broker single-user-mode flow; surfacing schema drift
    here as a clear error is preferable to handing the supervisor a
    silently-empty key."""

    def test_extracts_api_key(self) -> None:
        body = json.dumps({"apiKey": "oc_abc123"})
        assert _parse_api_key_response(body, source="GET") == "oc_abc123"

    def test_non_json_body_raises(self) -> None:
        with pytest.raises(BundledBrokerError, match="non-JSON"):
            _parse_api_key_response("not json", source="GET")

    def test_missing_field_raises(self) -> None:
        body = json.dumps({"otherField": "x"})
        with pytest.raises(BundledBrokerError, match="apiKey"):
            _parse_api_key_response(body, source="POST")

    def test_empty_api_key_raises(self) -> None:
        body = json.dumps({"apiKey": ""})
        with pytest.raises(BundledBrokerError, match="empty"):
            _parse_api_key_response(body, source="GET")


# ---------------------------------------------------------------------------
# Supervisor lifecycle
# ---------------------------------------------------------------------------


class TestSupervisorStart:
    """End-to-end bring-up via the fake subprocess + httpx transport.

    These are the load-bearing tests: they pin down the order of
    operations (compose up -> port discovery -> health -> key) and
    the BrokerConfig the supervisor synthesises for the rest of the
    gateway to consume.
    """

    @pytest.mark.asyncio
    async def test_happy_path_returns_external_broker_config(self) -> None:
        recorder = _ComposeRecorder()
        recorder.script("up")
        recorder.script("port", stdout="127.0.0.1:34567\n")
        recorder.script("port", stdout="127.0.0.1:34568\n")

        handler = _ok_health_handler(
            [(200, {"apiKey": "oc_existing"})],
        )
        supervisor = _make_supervisor(recorder=recorder, handler=handler)

        config = await supervisor.start()

        # The synthesised config carries ``mode="bundled"``; the
        # admin key is exposed separately via the supervisor's
        # ``admin_api_key`` attribute (in-process only -- never
        # written to disk and never exposed via an env var name on
        # the config object).
        assert config.mode == "bundled"
        assert config.enabled is True
        assert config.admin_url == "http://127.0.0.1:34567"
        assert config.proxy_url == "http://onecli:10255"
        assert config.admin_api_key_env_var is None
        # The literal admin key is held in process memory on the
        # supervisor, not on the config.
        assert supervisor.admin_api_key is not None
        assert str(supervisor.admin_api_key) == "oc_existing"

        # ``project_name`` is populated post-start for the gateway
        # to derive ``egress_network_name`` from.
        assert supervisor.project_name is not None
        assert supervisor.project_name.startswith("thorn-broker-")
        assert supervisor.egress_network_name == (
            f"{supervisor.project_name}_thorn-broker"
        )

        # Order-of-operations check: up first, then two ports, no down.
        verbs = [_ComposeRecorder._verb_of(call[0]) for call in recorder.calls]
        assert verbs == ["up", "port", "port"]

        # The compose env carries the supervisor's overrides; in
        # particular, ``ONECLI_NEXTAUTH_SECRET=""`` is what keeps OneCLI
        # in single-user mode (and therefore lets the unauthenticated
        # admin-key endpoint work).
        env = recorder.calls[0][1]
        assert env["ONECLI_ADMIN_PORT"] == "0"
        assert env["ONECLI_PROXY_PORT"] == "0"
        assert env["ONECLI_BIND_HOST"] == "127.0.0.1"
        assert env["ONECLI_NEXTAUTH_SECRET"] == ""

    @pytest.mark.asyncio
    async def test_mints_key_when_get_returns_404(self) -> None:
        # The mint-on-404 fallback is the clean path when OneCLI has
        # never had a key issued (fresh single-user-mode boot, which
        # is exactly the bundled-broker shape).
        recorder = _ComposeRecorder()
        recorder.script("up")
        recorder.script("port", stdout="127.0.0.1:1111\n")
        recorder.script("port", stdout="127.0.0.1:2222\n")

        handler = _ok_health_handler([
            (404, {"error": "not found"}),
            (200, {"apiKey": "oc_minted"}),
        ])
        supervisor = _make_supervisor(recorder=recorder, handler=handler)

        config = await supervisor.start()
        assert config.mode == "bundled"
        assert supervisor.admin_api_key is not None
        assert str(supervisor.admin_api_key) == "oc_minted"

    @pytest.mark.asyncio
    async def test_unexpected_status_on_get_raises(self) -> None:
        recorder = _ComposeRecorder()
        recorder.script("up")
        recorder.script("port", stdout="127.0.0.1:1111\n")
        recorder.script("port", stdout="127.0.0.1:2222\n")
        # On unexpected status, the supervisor must roll back the
        # stack so we don't leak compose state.
        recorder.script("down")

        handler = _ok_health_handler([(500, "internal error")])
        supervisor = _make_supervisor(recorder=recorder, handler=handler)

        with pytest.raises(BundledBrokerError, match="HTTP 500"):
            await supervisor.start()

        # Verify rollback fired.
        verbs = [_ComposeRecorder._verb_of(call[0]) for call in recorder.calls]
        assert verbs[-1] == "down"

    @pytest.mark.asyncio
    async def test_health_timeout_rolls_back(self) -> None:
        recorder = _ComposeRecorder()
        recorder.script("up")
        recorder.script("port", stdout="127.0.0.1:1111\n")
        recorder.script("port", stdout="127.0.0.1:2222\n")
        recorder.script("down")

        # Health check that always fails so the supervisor exhausts
        # its (very short) budget and propagates BundledBrokerError.
        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/health":
                return httpx.Response(500, text="not ready")
            raise AssertionError(
                f"unexpected URL during failing health: {request.url}",
            )

        supervisor = _make_supervisor(
            recorder=recorder,
            handler=httpx.MockTransport(_handler),
        )

        with pytest.raises(BundledBrokerError, match="/api/health"):
            await supervisor.start()

        verbs = [_ComposeRecorder._verb_of(call[0]) for call in recorder.calls]
        assert verbs[-1] == "down"

    @pytest.mark.asyncio
    async def test_start_is_single_use(self) -> None:
        recorder = _ComposeRecorder()
        recorder.script("up")
        recorder.script("port", stdout="127.0.0.1:1111\n")
        recorder.script("port", stdout="127.0.0.1:2222\n")

        handler = _ok_health_handler([(200, {"apiKey": "oc_x"})])
        supervisor = _make_supervisor(recorder=recorder, handler=handler)

        await supervisor.start()
        # A second start() on the same instance is a programming
        # error (each per-process bring-up gets its own supervisor).
        with pytest.raises(RuntimeError, match="only be called once"):
            await supervisor.start()


class TestSupervisorShutdown:
    """``shutdown()`` runs ``compose down --volumes`` so no broker
    state survives the gateway exit -- the whole point of the
    transient-by-design model in the plan."""

    @pytest.mark.asyncio
    async def test_shutdown_runs_compose_down_with_volumes(self) -> None:
        recorder = _ComposeRecorder()
        recorder.script("up")
        recorder.script("port", stdout="127.0.0.1:1111\n")
        recorder.script("port", stdout="127.0.0.1:2222\n")
        recorder.script("down")

        handler = _ok_health_handler([(200, {"apiKey": "oc_x"})])
        supervisor = _make_supervisor(recorder=recorder, handler=handler)

        await supervisor.start()
        await supervisor.shutdown()

        # Find the ``down`` invocation and confirm the rollback flags.
        down_calls = [
            call for call in recorder.calls
            if _ComposeRecorder._verb_of(call[0]) == "down"
        ]
        assert len(down_calls) == 1
        argv, _, _ = down_calls[0]
        # ``--volumes`` is the bit that ensures no broker state
        # survives.  ``--remove-orphans`` is belt-and-braces for the
        # case where a previous bring-up half-succeeded.
        assert "--volumes" in argv
        assert "--remove-orphans" in argv

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self) -> None:
        recorder = _ComposeRecorder()
        recorder.script("up")
        recorder.script("port", stdout="127.0.0.1:1111\n")
        recorder.script("port", stdout="127.0.0.1:2222\n")
        recorder.script("down")

        handler = _ok_health_handler([(200, {"apiKey": "oc_x"})])
        supervisor = _make_supervisor(recorder=recorder, handler=handler)

        await supervisor.start()
        await supervisor.shutdown()
        # A second shutdown must not call ``compose down`` again -- the
        # supervisor is the single point of authority for stack
        # lifecycle and re-running ``down`` could surface a confusing
        # "no such project" error from the OCI runtime.
        await supervisor.shutdown()

        down_count = sum(
            1 for call in recorder.calls
            if _ComposeRecorder._verb_of(call[0]) == "down"
        )
        assert down_count == 1

    @pytest.mark.asyncio
    async def test_shutdown_swallows_compose_failure(self) -> None:
        # A hung / failing ``compose down`` must not block the gateway
        # from exiting; the supervisor logs but does not propagate.
        recorder = _ComposeRecorder()
        recorder.script("up")
        recorder.script("port", stdout="127.0.0.1:1111\n")
        recorder.script("port", stdout="127.0.0.1:2222\n")
        recorder.script("down", rc=1, stderr="broken")

        handler = _ok_health_handler([(200, {"apiKey": "oc_x"})])
        supervisor = _make_supervisor(recorder=recorder, handler=handler)

        await supervisor.start()
        # Should not raise even though down returned non-zero.
        await supervisor.shutdown()
