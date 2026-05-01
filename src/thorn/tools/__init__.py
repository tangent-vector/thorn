"""Toolset bundles for thorn agents.

This package provides both the built-in file/search tools (re-exported from
``thorn.core._tools``) and domain-specific toolsets as submodules:

- ``thorn.tools.forge``  -- Unified forge abstraction (GitLab + GitHub)
- ``thorn.tools.github`` -- GitHub API client wrapper (requires ``thorn[github]``)
- ``thorn.tools.gitlab`` -- GitLab API client wrapper (requires ``thorn[gitlab]``)
- ``thorn.tools.peers``  -- Peer-registry lookup tools

Built-in tools are available directly::

    from thorn.tools import read_file, edit_file, create_file, list_directory

Domain-specific toolsets are imported as submodules::

    from thorn.tools import forge    # unified forge abstraction
    from thorn.tools import github   # requires PyGithub
    from thorn.tools import gitlab   # requires python-gitlab
    from thorn.tools import peers

Git operations no longer have a dedicated tool surface: agents drive
``git`` (and ``gh`` / ``glab``, where the operator has provisioned
them) via :func:`thorn.core._tools.run_shell` inside the sandbox.
Maintaining a parallel Python wrapper for ``git`` got us nothing
beyond a second API to keep in sync with the real binary, and any
operation an agent might want to do is already expressible as the
same shell command a human collaborator would type.

``run_shell`` is intentionally omitted from the default exports.
Import it explicitly from ``thorn.core._tools`` if you need it::

    from thorn.core._tools import run_shell

``write_file`` is deprecated in favour of ``edit_file`` + ``create_file``
but remains importable from ``thorn.core._tools`` for backward compatibility.
"""

from thorn.core._tools import (
    ALL_BUILTIN_TOOLS,
    FILE_READING,
    FILE_WRITING,
    FileEdit,
    create_file,
    delete_file,
    edit_file,
    find_files,
    list_directory,
    move_file,
    read_file,
    search_files,
    write_file,
)

__all__ = [
    "read_file",
    "edit_file",
    "create_file",
    "delete_file",
    "move_file",
    "write_file",
    "FileEdit",
    "list_directory",
    "find_files",
    "search_files",
    "ALL_BUILTIN_TOOLS",
    "FILE_READING",
    "FILE_WRITING",
]
