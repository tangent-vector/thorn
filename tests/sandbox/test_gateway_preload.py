"""Tests for the gateway's eager sandbox-executor preload (Phase B).

Verifies that ``Gateway._startup`` calls each sandbox executor's
``start()`` so cold-start failures (e.g. missing image) surface on
the gateway's own console rather than on the first inbound event.

We stub out the executor itself; the deeper "did the
ContainerDaemonHost call ``adapter.run`` correctly?" question is
covered by ``tests/sandbox/test_container.py`` and
``tests/sandbox/test_runtime_wiring.py``.  This file is solely
about the gateway's preload behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from thorn.core._agent import Agent
from thorn.gateway._gateway import Gateway
from thorn.runtime import Runtime
from thorn.runtime._paths import AgencyPaths
from thorn.runtime._session import AgentID
from thorn.sandbox import SandboxImageMissingError


class _StubProvider:
    pass


class _StubExecutor:
    """Minimal stand-in for :class:`DaemonToolExecutor`.

    Records ``start()``/``aclose()`` calls and optionally raises a
    pre-baked exception from ``start`` so we can simulate cold-start
    failures (e.g. ``SandboxImageMissingError``).
    """

    def __init__(self, *, raise_on_start: BaseException | None = None) -> None:
        self.started = False
        self.closed = False
        self._raise = raise_on_start
        self.host = None

    async def start(self) -> None:
        self.started = True
        if self._raise is not None:
            raise self._raise

    async def aclose(self) -> None:
        self.closed = True

    async def cancel(self, call_id: str) -> None:  # pragma: no cover - unused
        pass


def _make_runtime(tmp_path: Path) -> Runtime:
    paths = AgencyPaths(
        home_root=tmp_path / "home",
        workspace_root=tmp_path / "ws",
    )
    paths.home_root.mkdir(parents=True, exist_ok=True)
    paths.workspace_root.mkdir(parents=True, exist_ok=True)
    return Runtime(
        provider=_StubProvider(),  # type: ignore[arg-type]
        workspace_root=paths.workspace_root,
        paths=paths,
        sandbox_executor_enabled=True,
    )


def _persist_agent(runtime: Runtime, aid: str) -> Agent:
    agent = Agent(id=AgentID(aid), name=aid)
    runtime.save_agent(agent)
    return agent


def _patch_executor_factory(
    runtime: Runtime, executors: dict[str, _StubExecutor],
) -> None:
    """Replace ``Runtime._build_sandbox_executor`` with a stub factory.

    Returns ``None`` for agents not in ``executors``, otherwise the
    pre-built stub for that agent.  Mirrors the real factory's
    signature.
    """

    def _build(agent: Agent) -> Any:
        if agent.id is None:
            return None
        return executors.get(str(agent.id))

    runtime._build_sandbox_executor = _build  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_preload_starts_every_agents_executor(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    _persist_agent(runtime, "a1")
    _persist_agent(runtime, "a2")
    execs = {"a1": _StubExecutor(), "a2": _StubExecutor()}
    _patch_executor_factory(runtime, execs)

    gateway = Gateway(runtime=runtime, sources=[])
    async with runtime:
        await gateway._startup()

    assert execs["a1"].started
    assert execs["a2"].started


@pytest.mark.asyncio
async def test_preload_raises_sandbox_image_missing(tmp_path: Path) -> None:
    """A missing image should hard-fail ``Gateway._startup``."""
    runtime = _make_runtime(tmp_path)
    _persist_agent(runtime, "a1")
    err = SandboxImageMissingError(
        "sandbox image 'ghost:1' is not present in the local podman cache. "
        "Run `thorn sandbox build --tag ghost:1` ..."
    )
    execs = {"a1": _StubExecutor(raise_on_start=err)}
    _patch_executor_factory(runtime, execs)

    gateway = Gateway(runtime=runtime, sources=[])
    async with runtime:
        with pytest.raises(SandboxImageMissingError) as exc_info:
            await gateway._startup()
    assert "ghost:1" in str(exc_info.value)
    assert "thorn sandbox build" in str(exc_info.value)


@pytest.mark.asyncio
async def test_preload_with_no_agents_is_a_noop(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    gateway = Gateway(runtime=runtime, sources=[])
    async with runtime:
        await gateway._startup()


@pytest.mark.asyncio
async def test_planned_egress_allowlist_warning_when_set(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Phase D: the gateway emits a single warning at startup when
    ``sandbox.planned_egress_allowlist`` is non-empty, since the
    entries are only future operator intent."""
    import logging

    from thorn.gateway._config import (
        PlannedEgressAllowlistEntry,
        SandboxConfig,
    )

    paths = AgencyPaths(
        home_root=tmp_path / "home",
        workspace_root=tmp_path / "ws",
    )
    paths.home_root.mkdir(parents=True, exist_ok=True)
    paths.workspace_root.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(
        provider=_StubProvider(),  # type: ignore[arg-type]
        workspace_root=paths.workspace_root,
        paths=paths,
        sandbox_executor_enabled=True,
        sandbox_config=SandboxConfig(
            backend="subprocess",
            planned_egress_allowlist=[
                PlannedEgressAllowlistEntry(
                    host="status.internal", port=443,
                ),
            ],
        ),
    )

    gateway = Gateway(runtime=runtime, sources=[])
    caplog.set_level(logging.WARNING, logger="thorn.gateway._gateway")
    async with runtime:
        await gateway._startup()

    matching = [
        rec for rec in caplog.records
        if "planned_egress_allowlist" in rec.getMessage()
    ]
    assert len(matching) == 1, (
        f"expected exactly one planned_egress_allowlist warning, got "
        f"{[r.getMessage() for r in matching]}"
    )
    assert "status.internal:443" in matching[0].getMessage()
    assert "R3" in matching[0].getMessage()
    assert "no runtime effect" in matching[0].getMessage()


@pytest.mark.asyncio
async def test_planned_egress_allowlist_silent_when_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty planned allow-list (the default) emits no warning."""
    import logging

    from thorn.gateway._config import SandboxConfig

    paths = AgencyPaths(
        home_root=tmp_path / "home",
        workspace_root=tmp_path / "ws",
    )
    paths.home_root.mkdir(parents=True, exist_ok=True)
    paths.workspace_root.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(
        provider=_StubProvider(),  # type: ignore[arg-type]
        workspace_root=paths.workspace_root,
        paths=paths,
        sandbox_executor_enabled=True,
        sandbox_config=SandboxConfig(backend="subprocess"),
    )

    gateway = Gateway(runtime=runtime, sources=[])
    caplog.set_level(logging.WARNING, logger="thorn.gateway._gateway")
    async with runtime:
        await gateway._startup()

    assert not [
        rec for rec in caplog.records
        if "planned_egress_allowlist" in rec.getMessage()
    ]
