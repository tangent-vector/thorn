"""Gateway resource files shipped with the wheel.

This package exists so :mod:`importlib.resources` can locate the
bundled OneCLI broker ``broker.compose.yml`` regardless of whether
Thorn is installed from a source checkout (``uv sync``) or as a built
wheel.

Resources here are read-only data files, not Python modules.  The
``__init__.py`` is empty by design; locating files goes through
``importlib.resources.files(__name__)``.
"""

from __future__ import annotations
