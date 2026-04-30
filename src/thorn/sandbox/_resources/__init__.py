"""Sandbox resource files shipped with the wheel.

This package exists so :mod:`importlib.resources` can locate the
sandbox ``Dockerfile.sandbox`` regardless of whether thorn is
installed editably (``pip install -e .``) or as a built wheel.

Resources here are read-only data files, not Python modules.  The
``__init__.py`` is empty by design; locating files goes through
``importlib.resources.files(__name__)``.
"""

from __future__ import annotations
