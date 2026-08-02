"""Context-directory enumeration: phase 1 of the per-prompt pipeline.

Given the directory roots that bound a session's context-gathering
walk, produce a single list of :class:`ContextDirectory` entries
ordered from outer-most to inner-most.  This is **pure path logic**:
the function performs no filesystem access and never raises for
missing-on-disk directories.  Whether any given path actually exists
is the per-directory loaders' problem (phase 2,
:mod:`thorn.runtime._context_layers`).

The shipped pipeline (see ``docs/context-gathering.md``) identifies
three layers, in outer-to-inner order:

1. **operator** -- ``<agency-home>/agents/<agent-id>/``: a single
   directory, outside the agent's sandbox, where human operators
   inject must-include policy.  Optional; gateway populates it,
   CLI may or may not.

2. **agent home** -- the chain from the agent's home directory down
   into the *session-key home* directory
   (``<agent-home>/<session-key-as-path>``).  Each ancestor on the
   walk is a candidate context directory, supporting topical memory
   and skills organised under the agent's home tree.  In gateway
   mode the agent home is real; in CLI mode it is the agent's
   home-mount path under ``~/.thorn/agents/local/home/``.

3. **agent workspace** -- the chain from the *logical* agent
   workspace path down into the session workspace.  In gateway mode
   the logical agent workspace is the per-agent workspace mount; in
   CLI mode it is whatever
   :func:`thorn.runtime._project_detection.pick_logical_agent_workspace_path_for_cli_session`
   returned at startup.

Within each layer we walk inclusively from the outer bound to the
inner bound (parent-chain), so if ``outer == inner`` we get a single
entry, and otherwise every directory in between is included.  Across
layers we deduplicate by path, keeping the **outer-most** kind
assignment when the same path could plausibly appear under more than
one layer (a defensive pass; in practice the layers don't overlap).

The output is intentionally a flat ``list[ContextDirectory]`` rather
than a structure-of-arrays: per the design doc, every downstream
phase iterates this list once and produces its own list-of-bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ContextDirectoryKind(str, Enum):
    """Provenance tag for a :class:`ContextDirectory`.

    The kind identifies which layer of the per-session walk a
    directory came from, so per-category collectors in phase 2 can
    apply kind-conditional file policy (e.g. ``MEMORY.md`` is loaded
    from ``OPERATOR`` and ``AGENT_HOME`` directories but not from
    ``AGENT_WORKSPACE`` ones).

    The string values are stable -- they appear in serialised
    diagnostics and (eventually) in trace events -- so changing them
    is a breaking change to anything that consumes those.
    """

    OPERATOR = "operator"
    """``<agency-home>/agents/<agent-id>/`` -- operator policy
    injection point, outside the agent's sandbox."""

    AGENT_HOME = "agent_home"
    """A directory on the chain from the agent's home down to the
    session-key home.  The agent's "skull" -- private workspace for
    persistent state."""

    AGENT_WORKSPACE = "agent_workspace"
    """A directory on the chain from the logical agent workspace
    down to the session workspace.  The agent's "desk" -- the place
    where tool-driven work happens."""


@dataclass(frozen=True)
class ContextDirectory:
    """A single directory considered as a source of session context.

    The path is *not* required to exist on disk: phase 1 produces
    purely logical paths, and phase 2 filters out the empty ones.

    The kind tag drives the per-collector kind-filtering policy in
    phase 2.  Adding a new kind is a one-place change here followed
    by per-collector updates in :mod:`thorn.runtime._context_layers`.
    """

    path: Path
    kind: ContextDirectoryKind


# ---------------------------------------------------------------------------
# Layer walks
# ---------------------------------------------------------------------------

def _path_chain_inclusive(outer: Path, inner: Path) -> list[Path]:
    """Return every path from *outer* (first) down to *inner* (last).

    The walk is inclusive of both endpoints.  When ``outer == inner``
    the result is a single-element list ``[outer]``.

    The expected case is that *outer* is an ancestor of *inner*; the
    walk then traverses the parent chain.  As a defensive fallback
    for the documented exceptional case (the design doc's "if we
    have an agent workspace that doesn't contain the session
    workspace, but I suppose we need a plan in case that happens"),
    when *outer* does not enclose *inner* the result is the
    two-element list ``[outer, inner]`` -- both directories surface
    individually as separate context layers, with no attempt to
    bridge them by walking through unrelated parents.
    """
    if outer == inner:
        return [outer]
    if not _is_strict_ancestor(outer, inner):
        return [outer, inner]
    chain: list[Path] = [inner]
    cursor = inner
    while cursor != outer:
        cursor = cursor.parent
        chain.append(cursor)
    chain.reverse()
    return chain


def _is_strict_ancestor(maybe_ancestor: Path, descendant: Path) -> bool:
    """``True`` when *maybe_ancestor* is a proper ancestor of *descendant*.

    Equivalent to ``descendant.is_relative_to(maybe_ancestor)
    and descendant != maybe_ancestor`` but guards against the
    Python 3.9 ``ValueError`` shape of :meth:`Path.relative_to` on
    non-related paths -- which on 3.12 has been replaced with a
    boolean :meth:`Path.is_relative_to`, but the defensive shape
    keeps the code obvious.
    """
    if maybe_ancestor == descendant:
        return False
    try:
        return descendant.is_relative_to(maybe_ancestor)
    except ValueError:
        return False


def _resolved_layer_chain(
    *,
    outer: Path | None,
    inner: Path | None,
) -> list[Path]:
    """Resolve the (outer, inner) pair into an outer-to-inner path list.

    The collector takes a permissive view of partial inputs:

    - both ``None`` -> empty list (the layer contributes nothing);
    - only one provided -> a single-element list with that path
      (the present bound serves as both ends);
    - both provided -> the inclusive chain from *outer* to *inner*.

    This is the small bit of policy that lets callers pass the
    operator dir without an inner bound, or pass an agent-home walk
    without a session-key home, without scattering ``if x is None``
    branches through the main ``gather_context_directories`` body.
    """
    if outer is None and inner is None:
        return []
    if outer is None:
        assert inner is not None
        return [inner]
    if inner is None:
        return [outer]
    return _path_chain_inclusive(outer, inner)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def gather_context_directories(
    *,
    operator_dir: Path | None = None,
    agent_home_path: Path | None = None,
    session_key_home_path: Path | None = None,
    logical_agent_workspace_path: Path | None = None,
    session_workspace_path: Path | None = None,
) -> list[ContextDirectory]:
    """Build the outer-to-inner directory list for a session prompt.

    Three optional layers are concatenated in this order:

    1. **operator** -- a single ``OPERATOR`` entry for *operator_dir*,
       if given.  Typically the agent's framework dir
       (``<agency-home>/agents/<agent-id>/``); the per-prompt walk
       sees it but the agent's sandbox does not.

    2. **agent home** -- the inclusive chain from *agent_home_path*
       (outer) down to *session_key_home_path* (inner), each tagged
       ``AGENT_HOME``.  When only one of the pair is given, that
       single path becomes the layer's sole entry.  When both are
       ``None`` the layer is skipped entirely.

    3. **agent workspace** -- the inclusive chain from
       *logical_agent_workspace_path* (outer) down to
       *session_workspace_path* (inner), each tagged
       ``AGENT_WORKSPACE``.  Same partial-input policy as the home
       layer.

    A final dedup pass removes any path that already appeared
    earlier in the list, keeping the **outer-most** occurrence (and
    therefore the outer-most kind tag).  In practice the three
    layers root at distinct subtrees and don't overlap; the dedup
    pass is a defensive guard against pathological caller inputs.

    All path inputs are treated as **purely logical**: this
    function performs no ``stat`` or ``exists`` checks, raises
    nothing for missing-on-disk paths, and does not resolve
    symlinks.  The per-directory loaders in phase 2 are responsible
    for handling the absent-file case.
    """
    raw: list[ContextDirectory] = []

    if operator_dir is not None:
        raw.append(ContextDirectory(
            path=operator_dir,
            kind=ContextDirectoryKind.OPERATOR,
        ))

    home_chain = _resolved_layer_chain(
        outer=agent_home_path,
        inner=session_key_home_path,
    )
    for path in home_chain:
        raw.append(ContextDirectory(
            path=path,
            kind=ContextDirectoryKind.AGENT_HOME,
        ))

    workspace_chain = _resolved_layer_chain(
        outer=logical_agent_workspace_path,
        inner=session_workspace_path,
    )
    for path in workspace_chain:
        raw.append(ContextDirectory(
            path=path,
            kind=ContextDirectoryKind.AGENT_WORKSPACE,
        ))

    return _deduplicated_outer_first(raw)


def _deduplicated_outer_first(
    entries: list[ContextDirectory],
) -> list[ContextDirectory]:
    """Drop later occurrences of any path already seen, preserving order.

    The *first* occurrence wins -- which, given the outer-first
    layer ordering in :func:`gather_context_directories`, means the
    outer-most kind tag wins for any path that would otherwise
    appear twice.  Comparison is by exact path equality (no resolve,
    no normalization beyond what :class:`pathlib.PurePath` already
    provides) so callers must pass already-normalised paths if they
    need stricter dedup.
    """
    seen: set[Path] = set()
    out: list[ContextDirectory] = []
    for entry in entries:
        if entry.path in seen:
            continue
        seen.add(entry.path)
        out.append(entry)
    return out


__all__ = [
    "ContextDirectory",
    "ContextDirectoryKind",
    "gather_context_directories",
]
