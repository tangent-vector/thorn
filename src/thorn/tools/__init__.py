"""Toolset bundles for thorn agents.

This package provides both the built-in file/search tools (re-exported from
``thorn.core._tools``) and domain-specific toolsets as submodules:

- ``thorn.tools.git``    -- Git subprocess operations
- ``thorn.tools.github`` -- GitHub API operations (requires ``thorn[github]``)
- ``thorn.tools.gitlab`` -- GitLab API operations (requires ``thorn[gitlab]``)

Built-in tools are available directly::

    from thorn.tools import read_file, edit_file, create_file, list_directory

Domain-specific toolsets are imported as submodules::

    from thorn.tools import git
    from thorn.tools import github   # requires PyGithub
    from thorn.tools import gitlab   # requires python-gitlab

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
    ask_user,
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
    "ask_user",
    "ALL_BUILTIN_TOOLS",
    "FILE_READING",
    "FILE_WRITING",
]
