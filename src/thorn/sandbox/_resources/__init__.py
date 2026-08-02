"""Sandbox resource files shipped with the wheel.

This package exists so :mod:`importlib.resources` can locate the
sandbox resource files (``Dockerfile.sandbox``,
``thorn-sandbox-entrypoint``) regardless of whether thorn is
installed from a source checkout (``uv sync``) or as a built wheel.

Resources here are read-only data files, not Python modules.  The
``__init__.py`` carries only docstring + future declarations; locating
files goes through ``importlib.resources.files(__name__)``.
"""

from __future__ import annotations
