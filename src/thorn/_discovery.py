""".thorn/ directory discovery.

Walks from CWD up to the filesystem root (and checks the user home
directory) looking for ``.thorn/`` directories.  Python files inside
those directories are imported, and any functions decorated with
``@tool`` or ``@skill`` are collected for use as agent tools.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


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


def load_module_tools(path: Path) -> list[Callable[..., Any]]:
    """Import a ``.py`` file and return functions marked with ``@tool`` or ``@skill``.

    Import errors (syntax errors, missing dependencies, etc.) are logged
    as warnings rather than propagated — a broken file in ``.thorn/``
    should not prevent the rest of discovery from working.
    """
    module_name = f"_thorn_user_.{path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("could not create import spec for %s", path)
            return []
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        logger.warning("failed to import %s", path, exc_info=True)
        return []
    finally:
        sys.modules.pop(module_name, None)

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


def discover_tools(start: Path | None = None) -> list[Callable[..., Any]]:
    """Find all ``@tool`` and ``@skill`` functions in ``.thorn/`` directories.

    Combines :func:`find_thorn_dirs` and :func:`load_module_tools` to
    produce a flat list of callables ready for ``_prepare_tools()``.
    """
    result: list[Callable[..., Any]] = []
    seen_names: set[str] = set()

    for thorn_dir in find_thorn_dirs(start):
        for py_file in sorted(thorn_dir.glob("*.py")):
            for fn in load_module_tools(py_file):
                name = getattr(fn, "__name__", str(fn))
                if name not in seen_names:
                    result.append(fn)
                    seen_names.add(name)
                else:
                    logger.debug(
                        "skipping duplicate tool %r from %s", name, py_file,
                    )
    return result
