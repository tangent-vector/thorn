""".thorn/ directory discovery.

Walks from CWD up to the filesystem root (and checks the user home
directory) looking for ``.thorn/`` directories.  Python files inside
those directories are imported, and any functions decorated with
``@tool`` or ``@skill`` are collected for use as agent tools.

Each ``.thorn/`` directory is registered as a synthetic Python package
so that files within it can use relative imports to reference siblings::

    # In .thorn/dev_tools.py
    from .build_tools import build, run_calc
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
# .thorn/ directory location
# ---------------------------------------------------------------------------

def find_thorn_dirs(start: Path | None = None) -> list[Path]:
    """Locate ``.thorn/`` directories by walking up from *start*.

    Returns directories ordered deepest-first (most specific first).
    The user home directory (``~/.thorn``) is appended last, if it
    exists and wasn't already found during the walk.
    """
    if start is None:
        start = Path.cwd()
    start = start.resolve()

    found: list[Path] = []
    seen: set[Path] = set()

    current = start
    while True:
        candidate = current / ".thorn"
        if candidate.is_dir() and candidate not in seen:
            found.append(candidate)
            seen.add(candidate)
        parent = current.parent
        if parent == current:
            break
        current = parent

    home_thorn = Path.home() / ".thorn"
    if home_thorn.is_dir() and home_thorn.resolve() not in seen:
        found.append(home_thorn)

    return found


def load_workspace_instructions(workspace_root: Path) -> str | None:
    """Read ``AGENTS.md`` from *workspace_root*, if it exists.

    Returns the file contents as a string, or ``None`` when the file is
    absent or unreadable.
    """
    agents_md = workspace_root / "AGENTS.md"
    if not agents_md.is_file():
        return None
    try:
        return agents_md.read_text(encoding="utf-8")
    except OSError:
        logger.warning("failed to read %s", agents_md, exc_info=True)
        return None


def load_agent_memory(workspace: Path) -> str | None:
    """Read ``MEMORY.md`` from *workspace*, if it exists.

    ``MEMORY.md`` holds instance-specific knowledge for an agent (e.g.
    "the repository URL is X", "the default branch is Y").  It belongs
    to the agent *instance* (not the role/class) and is auto-injected
    into the system prompt when present.

    Returns the file contents as a string, or ``None`` when the file is
    absent or unreadable.
    """
    memory_md = workspace / "MEMORY.md"
    if not memory_md.is_file():
        return None
    try:
        return memory_md.read_text(encoding="utf-8")
    except OSError:
        logger.warning("failed to read %s", memory_md, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Synthetic package management for .thorn/ directories
# ---------------------------------------------------------------------------

def _package_name_for_dir(thorn_dir: Path) -> str:
    """Generate a stable synthetic package name for a ``.thorn/`` directory."""
    dir_hash = hashlib.sha256(str(thorn_dir.resolve()).encode()).hexdigest()[:12]
    return f"_thorn_user_.d{dir_hash}"


def _ensure_package(thorn_dir: Path) -> str:
    """Register *thorn_dir* as a synthetic Python package in ``sys.modules``.

    Returns the package name.  Repeated calls for the same directory
    are idempotent.  The package's ``__path__`` is set so that Python's
    standard ``PathFinder`` can resolve relative imports between sibling
    ``.py`` files in the directory.
    """
    thorn_dir = thorn_dir.resolve()
    pkg_name = _package_name_for_dir(thorn_dir)
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(thorn_dir)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
    return pkg_name


def _load_thorn_module(thorn_dir: Path, py_file: Path) -> types.ModuleType | None:
    """Load a ``.py`` file as a submodule of the synthetic package for *thorn_dir*.

    The module is kept in ``sys.modules`` so that sibling files can
    import it via relative imports (``from .sibling import thing``).

    Returns the loaded module, or ``None`` on failure.
    """
    pkg_name = _ensure_package(thorn_dir)
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

def load_module_tools(path: Path) -> list[Callable[..., Any]]:
    """Import a ``.py`` file and return functions marked with ``@tool`` or ``@skill``.

    The file is loaded as a submodule of a synthetic package for its
    parent directory, enabling relative imports between sibling files.

    Import errors (syntax errors, missing dependencies, etc.) are logged
    as warnings rather than propagated — a broken file in ``.thorn/``
    should not prevent the rest of discovery from working.
    """
    module = _load_thorn_module(path.parent, path)
    if module is None:
        return []
    return _scan_tools(module)


def discover_tools(start: Path | None = None) -> list[Callable[..., Any]]:
    """Find all ``@tool`` and ``@skill`` functions in ``.thorn/`` directories.

    Each ``.thorn/`` directory is registered as a synthetic package and
    all ``.py`` files within it are loaded before scanning for tools.
    This ensures sibling imports resolve correctly even when loading
    order would otherwise matter.
    """
    result: list[Callable[..., Any]] = []
    seen_names: set[str] = set()

    for thorn_dir in find_thorn_dirs(start):
        modules: list[types.ModuleType] = []
        for py_file in sorted(thorn_dir.glob("*.py")):
            module = _load_thorn_module(thorn_dir, py_file)
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
