"""``thorn.sandbox`` -- container-backed tool execution (Phase B).

Phase B wraps the Phase-A ``thorn-toolhost`` daemon in an OCI
container while keeping the brain on the host.  This package owns:

* :mod:`thorn.sandbox._runtime` -- the :class:`OCIRuntimeAdapter`
  protocol plus podman/docker/fake implementations.
* :mod:`thorn.sandbox._container` -- :class:`ContainerDaemonHost`,
  the :class:`~thorn.toolhost.DaemonHost` implementation that drives
  per-agent containers.
* :mod:`thorn.sandbox._image` -- image-presence helpers and the
  default ``thorn-sandbox:<version>`` tag derivation.
* :mod:`thorn.sandbox._config` -- the ``SandboxBackend`` config
  object plumbed through the CLI/runtime.

All container-runtime concerns live here so :mod:`thorn.toolhost`
stays neutral about how the daemon is launched.
"""

from __future__ import annotations

from thorn.sandbox._container import (
    CONTAINER_CONTROL_DIR,
    CONTAINER_HOME_DIR,
    CONTAINER_LOG_PATH,
    CONTAINER_RUNTIME_DIR,
    CONTAINER_SOCKET_PATH,
    CONTAINER_WORKSPACE_DIR,
    ContainerDaemonHost,
    ContainerHostConfig,
    ContainerNotReadyError,
    ContainerStartTimeoutError,
    SandboxImageMissingError,
    derive_container_name,
)
from thorn.sandbox._image import (
    DEFAULT_SANDBOX_DOCKERFILE,
    DEFAULT_SANDBOX_IMAGE_NAME,
    build_default_sandbox_image,
    default_sandbox_image_tag,
    ensure_sandbox_image,
    find_default_sandbox_dockerfile,
)
from thorn.sandbox._resolve import ResolvedSandboxConfig, resolve_sandbox_config
from thorn.sandbox._runtime import (
    ContainerSpec,
    ContainerState,
    DockerAdapter,
    FakeOCIRuntimeAdapter,
    Mount,
    OCIImageMissing,
    OCIRuntimeAdapter,
    OCIRuntimeError,
    OCIRuntimeNotFound,
    PodmanAdapter,
    Tmpfs,
    select_oci_runtime,
)

__all__ = [
    "CONTAINER_CONTROL_DIR",
    "CONTAINER_HOME_DIR",
    "CONTAINER_LOG_PATH",
    "CONTAINER_RUNTIME_DIR",
    "CONTAINER_SOCKET_PATH",
    "CONTAINER_WORKSPACE_DIR",
    "ContainerDaemonHost",
    "ContainerHostConfig",
    "ContainerNotReadyError",
    "ContainerSpec",
    "ContainerStartTimeoutError",
    "ContainerState",
    "DEFAULT_SANDBOX_DOCKERFILE",
    "DEFAULT_SANDBOX_IMAGE_NAME",
    "DockerAdapter",
    "FakeOCIRuntimeAdapter",
    "Mount",
    "OCIImageMissing",
    "OCIRuntimeAdapter",
    "OCIRuntimeError",
    "OCIRuntimeNotFound",
    "PodmanAdapter",
    "ResolvedSandboxConfig",
    "SandboxImageMissingError",
    "Tmpfs",
    "build_default_sandbox_image",
    "default_sandbox_image_tag",
    "derive_container_name",
    "ensure_sandbox_image",
    "find_default_sandbox_dockerfile",
    "resolve_sandbox_config",
    "select_oci_runtime",
]
