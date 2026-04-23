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
from pathlib import Path

from thorn.sandbox._container import SandboxImageMissingError
from thorn.sandbox._runtime import OCIRuntimeAdapter

logger = logging.getLogger(__name__)


DEFAULT_SANDBOX_IMAGE_NAME = "thorn-sandbox"
"""Unversioned image name shared by the Dockerfile and the CLI."""

DEFAULT_SANDBOX_DOCKERFILE = "Dockerfile.sandbox"
"""Filename of the in-tree sandbox Dockerfile."""


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


def find_default_sandbox_dockerfile(start: Path | None = None) -> Path:
    """Locate ``Dockerfile.sandbox`` shipped with this repository.

    Walks up from the directory holding the ``thorn`` source until it
    finds a sibling ``Dockerfile.sandbox``.  Used by ``thorn sandbox
    build`` so operators do not have to type the path; tests can
    override by passing ``--dockerfile``.

    Raises :class:`FileNotFoundError` when no ``Dockerfile.sandbox``
    is reachable from the framework's source location -- typical for
    pip-installed (non-editable) installs where only the wheel
    contents land on disk.
    """
    if start is None:
        from thorn import __file__ as thorn_file
        start = Path(thorn_file).resolve().parent

    for candidate in (start, *start.parents):
        dockerfile = candidate / DEFAULT_SANDBOX_DOCKERFILE
        if dockerfile.is_file():
            return dockerfile
    raise FileNotFoundError(
        f"Could not locate {DEFAULT_SANDBOX_DOCKERFILE} starting from "
        f"{start}; pass --dockerfile explicitly when running from a "
        "non-source-checkout install.",
    )


async def build_default_sandbox_image(
    adapter: OCIRuntimeAdapter,
    *,
    tag: str | None = None,
    dockerfile: Path | None = None,
    context: Path | None = None,
) -> str:
    """Build the in-tree sandbox image, returning the tag it was tagged with.

    *tag* defaults to :func:`default_sandbox_image_tag`; *dockerfile*
    defaults to :func:`find_default_sandbox_dockerfile`; *context*
    defaults to the directory containing the dockerfile (the
    repository root in the in-tree case).

    Logs (at INFO) the resolved tag and dockerfile path so the
    operator can confirm what was actually built without reaching for
    the runtime's own output.
    """
    resolved_tag = tag or default_sandbox_image_tag()
    resolved_dockerfile = dockerfile or find_default_sandbox_dockerfile()
    resolved_context = context or resolved_dockerfile.parent

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
