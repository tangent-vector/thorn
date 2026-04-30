"""Locate gateway resource files shipped with the wheel.

Right now the only wheel-shipped resource is the bundled OneCLI
broker compose YAML used by :class:`BundledBrokerSupervisor`.  The
helper lives in its own module rather than inside the supervisor so
non-supervisor callers (tests, ``thorn broker`` CLI helpers, future
operator-facing "dump the bundled compose" subcommand) can reach it
without taking on the supervisor's import cost.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Iterator

_RESOURCE_PACKAGE = "thorn.gateway._resources"
"""Package containing wheel-shipped gateway resource files."""

BUNDLED_BROKER_COMPOSE_FILENAME = "broker.compose.yml"
"""Filename of the bundled OneCLI broker compose YAML."""


def _packaged_broker_compose() -> Traversable:
    """Return the ``Traversable`` for the wheel-shipped broker compose.

    Raises :class:`FileNotFoundError` when the resource cannot be
    located -- a packaging defect rather than an operator-facing
    error, since the file is unconditionally shipped with every
    install of ``thorn``.
    """
    traversable = files(_RESOURCE_PACKAGE).joinpath(BUNDLED_BROKER_COMPOSE_FILENAME)
    if not traversable.is_file():
        raise FileNotFoundError(
            f"Bundled broker compose file {BUNDLED_BROKER_COMPOSE_FILENAME!r} "
            f"not found in resource package {_RESOURCE_PACKAGE!r}.  "
            "This indicates a packaging error in the installed `thorn` "
            "distribution.",
        )
    return traversable


@contextmanager
def materialize_bundled_broker_compose() -> Iterator[Path]:
    """Yield a real on-disk path for the bundled broker compose YAML.

    Wraps :func:`importlib.resources.as_file` so callers (notably
    ``BundledBrokerSupervisor``) can pass the path to ``docker
    compose -f`` regardless of whether the installed distribution
    keeps resources on the filesystem (editable installs, plain
    wheel) or inside a zipped wheel.

    The yielded :class:`~pathlib.Path` is only guaranteed to exist
    for the lifetime of the ``with`` block.  ``docker compose up
    -d`` reads the file synchronously during invocation, so by the
    time the context manager exits compose has already loaded the
    YAML; the in-memory project descriptor it tracks does not need
    the file to remain on disk.
    """
    traversable = _packaged_broker_compose()
    with as_file(traversable) as on_disk:
        yield Path(on_disk).resolve()


def read_bundled_broker_compose_text() -> str:
    """Return the bundled broker compose YAML as a UTF-8 string.

    Convenience for ``thorn broker`` debugging helpers and the
    operator-facing "extract this and run it yourself" workflow.
    Reads the resource directly via
    :func:`importlib.resources.files`, no temp file needed.
    """
    return _packaged_broker_compose().read_text(encoding="utf-8")


__all__ = [
    "BUNDLED_BROKER_COMPOSE_FILENAME",
    "materialize_bundled_broker_compose",
    "read_bundled_broker_compose_text",
]
