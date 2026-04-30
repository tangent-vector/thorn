"""Sandbox image helpers: name derivation, presence probe, build entrypoint.

Phase B's image story is "operator names a built image".  This module
provides the small surface that lets the rest of the system follow
that contract:

* :data:`DEFAULT_SANDBOX_IMAGE_NAME` -- the unversioned name shared
  by the in-tree Dockerfile and the CLI.
* :func:`default_sandbox_image_tag` -- ``thorn-sandbox:<version>``,
  derived from the installed ``thorn`` package version so a freshly
  built image lines up with the running framework without operator
  bookkeeping.
* :func:`ensure_sandbox_image` -- async presence check that raises
  the same :class:`SandboxImageMissingError` the host would, with a
  remediation message tailored to whichever side noticed the absence.
* :func:`build_default_sandbox_image` -- driver for ``thorn sandbox
  build``; locates ``Dockerfile.sandbox`` next to the source tree
  (or via an explicit override) and asks the configured OCI runtime
  to build it.

The CLI command itself lives in :mod:`thorn._cli`; this module owns
only logic that's useful from non-CLI callers (e.g. tests, a future
``thorn sandbox status`` that wants to compute the default tag).
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

from thorn.sandbox._container import SandboxImageMissingError
from thorn.sandbox._runtime import OCIRuntimeAdapter

logger = logging.getLogger(__name__)


DEFAULT_SANDBOX_IMAGE_NAME = "thorn-sandbox"
"""Unversioned image name shared by the Dockerfile and the CLI."""

DEFAULT_SANDBOX_DOCKERFILE = "Dockerfile.sandbox"
"""Filename of the in-tree sandbox Dockerfile."""

_RESOURCE_PACKAGE = "thorn.sandbox._resources"
"""Package containing the wheel-shipped Dockerfile.sandbox."""


def _thorn_version() -> str:
    """Return the installed ``thorn`` version, or ``"dev"`` if unknown.

    The ``"dev"`` fallback covers the "running from a source checkout
    without an installed package" case that comes up in tests; using
    a stable string keeps the default tag deterministic in that
    environment.
    """
    try:
        return version("thorn")
    except PackageNotFoundError:
        return "dev"


def default_sandbox_image_tag() -> str:
    """Return the default ``thorn-sandbox:<version>`` image tag.

    Tied to the installed thorn version so a fresh build of the
    sandbox image is implicitly versioned with the framework that
    will speak to it.  Operators can override per-agent via
    ``agent.json`` ``sandbox.image`` and agency-wide via
    ``gateway.json`` ``sandbox.image``.
    """
    return f"{DEFAULT_SANDBOX_IMAGE_NAME}:{_thorn_version()}"


async def ensure_sandbox_image(
    adapter: OCIRuntimeAdapter,
    image: str,
) -> None:
    """Raise :class:`SandboxImageMissingError` if *image* is not cached.

    Phase B's stance is to fail loudly rather than auto-build: the
    error message names the exact image and the exact CLI command to
    rebuild it, so the operator's response is mechanical.

    Used by the CLI ``thorn sandbox status`` command and by the
    gateway during agent pre-load so the failure surfaces *before*
    the first incoming event hits the agent.
    """
    if await adapter.image_exists(image):
        return
    raise SandboxImageMissingError(
        f"sandbox image {image!r} is not present in the local "
        f"{adapter.name} cache.  Run `thorn sandbox build "
        f"--tag {image}` (or omit --tag for the default), or set "
        f"`sandbox.image` in gateway.json to an image that has been "
        f"built/pulled, then restart the gateway.",
    )


def _packaged_sandbox_dockerfile() -> Traversable | None:
    """Return the ``Traversable`` for the wheel-shipped Dockerfile, if any.

    Resolves ``thorn.sandbox._resources/Dockerfile.sandbox`` via
    :mod:`importlib.resources`.  Returns ``None`` when the resource
    package cannot be located (developer paths that strip the
    ``_resources/`` directory) or the file is missing.
    """
    try:
        traversable = files(_RESOURCE_PACKAGE).joinpath(DEFAULT_SANDBOX_DOCKERFILE)
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    if not traversable.is_file():
        return None
    return traversable


def find_default_sandbox_dockerfile(start: Path | None = None) -> Path:
    """Locate ``Dockerfile.sandbox`` shipped with this distribution.

    Default resolution (``start`` is ``None``):

    1. The wheel-shipped resource at
       ``thorn/sandbox/_resources/Dockerfile.sandbox`` via
       :mod:`importlib.resources`.  This is the production path:
       editable installs see the file on disk; built-wheel installs
       see it after :mod:`importlib.resources` materialises it.
    2. A walk up from the ``thorn`` source root looking for a
       sibling ``Dockerfile.sandbox``.  Kept as a fallback for
       non-standard layouts where the resource directory has been
       removed.

    When *start* is explicitly given, only the parent-walk fallback
    runs (rooted at *start*).  The explicit-start shape is a test
    seam for "walk doesn't find anything"; production callers leave
    *start* unset and get the resource-first behavior.

    Raises :class:`FileNotFoundError` when no path resolves.

    Note: when the file is materialised from a zipped wheel, the
    returned path is a *temporary* extraction.  The caller is
    expected to consume it synchronously (for example, hand it to
    ``docker build`` and discard the path before the next call).
    Persistent storage of the path is not supported.
    """
    if start is None:
        packaged = _packaged_sandbox_dockerfile()
        if packaged is not None:
            with as_file(packaged) as on_disk:
                return Path(on_disk).resolve()
        from thorn import __file__ as thorn_file
        start = Path(thorn_file).resolve().parent

    for candidate in (start, *start.parents):
        dockerfile = candidate / DEFAULT_SANDBOX_DOCKERFILE
        if dockerfile.is_file():
            return dockerfile
    raise FileNotFoundError(
        f"Could not locate {DEFAULT_SANDBOX_DOCKERFILE} via the "
        "wheel-shipped resource path or by walking up from "
        f"{start}; pass --dockerfile explicitly when running from a "
        "non-standard install.",
    )


def _find_build_context_for_dockerfile(dockerfile: Path) -> Path:
    """Find the source-tree root (the directory holding ``pyproject.toml``).

    The bundled ``Dockerfile.sandbox`` does ``COPY pyproject.toml ./``
    and ``COPY src/ src/``, so the build context must be the
    repository / source-tree root rather than wherever the Dockerfile
    itself happens to live.  Walks up from *dockerfile*'s parent
    until a sibling ``pyproject.toml`` is found.

    Raises :class:`FileNotFoundError` when no ``pyproject.toml`` is
    reachable -- the typical signal that ``thorn sandbox build`` was
    invoked from a wheel install (where the source tree was discarded
    after install).  The error message names the right remediation
    (pull a published image, or run from a source checkout).
    """
    start = dockerfile.parent.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate a source-tree root with pyproject.toml by "
        f"walking up from {dockerfile}.  The bundled Dockerfile.sandbox "
        "needs the thorn source tree as its build context, which is "
        "not available in wheel-only installs.  Either run `thorn "
        "sandbox build` from a source checkout, or use a pre-built "
        "image (and pass `--context` explicitly if you really do want "
        "to build from a custom layout).",
    )


async def build_default_sandbox_image(
    adapter: OCIRuntimeAdapter,
    *,
    tag: str | None = None,
    dockerfile: Path | None = None,
    context: Path | None = None,
) -> str:
    """Build the bundled sandbox image, returning the tag it was tagged with.

    *tag* defaults to :func:`default_sandbox_image_tag`; *dockerfile*
    defaults to :func:`find_default_sandbox_dockerfile`; *context*
    defaults to :func:`_find_build_context_for_dockerfile` (the
    nearest ancestor of the dockerfile that contains
    ``pyproject.toml``).

    Logs (at INFO) the resolved tag, dockerfile, and context so the
    operator can confirm what was actually built without reaching for
    the runtime's own output.
    """
    resolved_tag = tag or default_sandbox_image_tag()
    resolved_dockerfile = dockerfile or find_default_sandbox_dockerfile()
    resolved_context = context or _find_build_context_for_dockerfile(
        resolved_dockerfile,
    )

    logger.info(
        "sandbox: building %s from %s (context=%s, runtime=%s)",
        resolved_tag, resolved_dockerfile, resolved_context, adapter.name,
    )
    await adapter.build(
        context=resolved_context,
        dockerfile=resolved_dockerfile,
        tag=resolved_tag,
    )
    return resolved_tag


__all__ = [
    "DEFAULT_SANDBOX_DOCKERFILE",
    "DEFAULT_SANDBOX_IMAGE_NAME",
    "SandboxImageMissingError",
    "build_default_sandbox_image",
    "default_sandbox_image_tag",
    "ensure_sandbox_image",
    "find_default_sandbox_dockerfile",
]
