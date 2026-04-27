"""Project-level Python tool discovery from ``.agents/thorn/``.

This module's sole remaining responsibility is the
``@tool`` / ``@skill`` Python-decorator-based tool discovery that
:func:`discover_tools` performs against ``.agents/thorn/*.py`` files.
Everything else that used to live here -- ``.thorn/`` directory
walking, ``AGENTS.md`` and ``MEMORY.md`` loading -- has been
absorbed into the unified per-prompt context-gathering pipeline
(``thorn.runtime._context_paths`` /
``thorn.runtime._context_layers``).

Files in an ``.agents/thorn/`` directory are imported, and any
functions decorated with ``@tool`` or ``@skill`` are collected for
use as agent tools.  Each such directory is registered as a
synthetic Python package so that files within it can use relative
imports::

    # In .agents/thorn/dev_tools.py
    from .build_tools import build, run_calc

This is the *Python-callable* tool entry point.  Markdown-based
agent skills (``.agents/skills/<name>/SKILL.md``) live in the new
context-gathering pipeline; the plan calls out unifying these two
under one umbrella as a follow-up (see ``TODO.md``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .agents/thorn/ directory location (project tool definitions)
# ---------------------------------------------------------------------------

def _find_agents_thorn_dirs(start: Path | None = None) -> list[Path]:
    """Locate ``.agents/thorn/`` directories by walking up from *start*.

    Returns directories ordered deepest-first (most specific first).
    Only directories that actually contain at least one ``.py`` file
    are returned.

    Internal helper for :func:`discover_tools`; not part of the
    public surface (the previous public name was retired alongside
    its cousins as part of the unified-context-gathering refactor).
    """
    if start is None:
        start = Path.cwd()
    start = start.resolve()

    found: list[Path] = []
    seen: set[Path] = set()

    current = start
    while True:
        agents_dir = current / ".agents"
        if agents_dir.is_dir():
            thorn_pkg = agents_dir / "thorn"
            if thorn_pkg.is_dir() and thorn_pkg.resolve() not in seen:
                if any(thorn_pkg.glob("*.py")):
                    found.append(thorn_pkg)
                    seen.add(thorn_pkg.resolve())
        parent = current.parent
        if parent == current:
            break
        current = parent

    return found


# ---------------------------------------------------------------------------
# Synthetic package management for tool directories
# ---------------------------------------------------------------------------

def _package_name_for_dir(tool_dir: Path) -> str:
    """Generate a stable synthetic package name for a tool directory."""
    dir_hash = hashlib.sha256(str(tool_dir.resolve()).encode()).hexdigest()[:12]
    return f"_thorn_user_.d{dir_hash}"


def _ensure_package(tool_dir: Path) -> str:
    """Register *tool_dir* as a synthetic Python package in ``sys.modules``.

    Returns the package name.  Repeated calls for the same directory
    are idempotent.  The package's ``__path__`` is set so that Python's
    standard ``PathFinder`` can resolve relative imports between sibling
    ``.py`` files in the directory.
    """
    tool_dir = tool_dir.resolve()
    pkg_name = _package_name_for_dir(tool_dir)
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(tool_dir)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
    return pkg_name


def _load_module(tool_dir: Path, py_file: Path) -> types.ModuleType | None:
    """Load a ``.py`` file as a submodule of the synthetic package for *tool_dir*.

    The module is kept in ``sys.modules`` so that sibling files can
    import it via relative imports (``from .sibling import thing``).

    Returns the loaded module, or ``None`` on failure.
    """
    pkg_name = _ensure_package(tool_dir)
    module_name = f"{pkg_name}.{py_file.stem}"

    if module_name in sys.modules:
        return sys.modules[module_name]

    try:
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            logger.warning("could not create import spec for %s", py_file)
            return None
        module = importlib.util.module_from_spec(spec)
        module.__package__ = pkg_name
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        logger.warning("failed to import %s", py_file, exc_info=True)
        sys.modules.pop(module_name, None)
        return None


def _scan_tools(module: types.ModuleType) -> list[Callable[..., Any]]:
    """Extract ``@tool`` and ``@skill`` decorated functions from *module*."""
    collected: list[Callable[..., Any]] = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name)
        if callable(obj) and (
            getattr(obj, "_thorn_tool", False)
            or getattr(obj, "_thorn_skill", False)
        ):
            collected.append(obj)
    return collected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_tools(start: Path | None = None) -> list[Callable[..., Any]]:
    """Find all ``@tool`` and ``@skill`` functions in ``.agents/thorn/`` directories.

    Each ``.agents/thorn/`` directory is registered as a synthetic
    package and all ``.py`` files within it are loaded before scanning
    for tools.  This ensures sibling imports resolve correctly even when
    loading order would otherwise matter.

    Tools are deduplicated by function name (first occurrence wins,
    deepest directory first).
    """
    result: list[Callable[..., Any]] = []
    seen_names: set[str] = set()

    for tool_dir in _find_agents_thorn_dirs(start):
        modules: list[types.ModuleType] = []
        for py_file in sorted(tool_dir.glob("*.py")):
            module = _load_module(tool_dir, py_file)
            if module is not None:
                modules.append(module)

        for module in modules:
            for fn in _scan_tools(module):
                name = getattr(fn, "__name__", str(fn))
                if name not in seen_names:
                    result.append(fn)
                    seen_names.add(name)
                else:
                    logger.debug(
                        "skipping duplicate tool %r from %s",
                        name, getattr(module, "__file__", "?"),
                    )
    return result


__all__ = ["discover_tools"]
