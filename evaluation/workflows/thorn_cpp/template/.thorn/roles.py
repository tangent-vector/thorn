"""Agent role definitions for the modular C++ development workflow.

Defines the role hierarchy:

- ``Developer`` (abstract): shared project knowledge and inspection tools
- ``ModuleDeveloper``: handles the full development lifecycle for a single
  module (architecture, API design, implementation, testing) and delegates
  sub-module work to child developers
- ``Concierge``: injects system prompts into the top-level thorn agent
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from thorn import Agent, FileAccessLevel, FileAccessRule, create_file, edit_file
from thorn.tools import FILE_READING

if TYPE_CHECKING:
    from thorn.core._context_injection import SeedContent

from .build_tools import build, run_tests
from .module_tools import (
    PROJECT_DIR,
    add_module,
    dependency_order,
    list_all_modules,
    list_submodules,
    module_header_path,
    module_source_path,
    module_status,
)
from .orchestration import delegate_to_child

# ---------------------------------------------------------------------------
# Shared system prompt fragments
# ---------------------------------------------------------------------------

_FILESYSTEM_CONVENTION = """\
The code lives under src/, and tests live under tests/.

The project is organized as a hierarchy of modules.
A top-level module `foo` comprises:

- A header file `src/foo.h`
- A source file `src/foo.cpp`
- Optionally, a test file `tests/foo_test.cpp`

A nested module `foo.bar` comprises:

- A header file `src/foo/bar.h`
- A source file `src/foo/bar.cpp`
- Optionally, a test file `tests/foo/bar_test.cpp`

If a module has children, they live in a subdirectory named after the module.
For example, children of `foo` live in `src/foo/`, and children of `foo.bar`
live in `src/foo/bar/`.

As a special case, the root module `main` comprises only the source file `src/main.cpp`.

Project-internal include paths are relative to src/.
For example, #include "foo/bar.h" includes the `foo.bar` module header.

If in doubt, use the module_header_path and module_source_path tools to resolve paths.
"""

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Developer(Agent, abstract=True):
    """Base for all workflow agents with shared project knowledge."""

    system_prompts = [
        (
            "You are a specialized agent working in the context of a "
            "software project. You handle incoming requests in accordance "
            "with your assigned role and scope of responsibility. You trust "
            "your teammates to handle their own responsibilities, and to "
            "have the same overall awareness of the codebase and its "
            "content as you do. "
        ),
        (
            "You must not attempt to write code or make other changes "
            "beyond what your assigned role allows. If you cannot complete "
            "your task within your allowed scope, without resorting to "
            "workarounds, you MUST call raise_error with a clear "
            "explanation rather than attempting work outside your mandate. "
        ),
        (
            "When reporting completion, be concise: just state what you "
            "created, modified, or verified. Only report information that "
            "will be useful to decision-making at higher levels of the "
            "project, and that cannot be inferred from the codebase itself. "
        ),
        _FILESYSTEM_CONVENTION,
    ]
    tools = [
        FILE_READING, list_submodules,
        module_header_path, module_source_path, list_all_modules,
    ]
    file_access = [
        FileAccessRule("**", FileAccessLevel.READ),
        FileAccessRule(".thorn", FileAccessLevel.HIDDEN),
        FileAccessRule(".thorn/**", FileAccessLevel.HIDDEN),
    ]


# ---------------------------------------------------------------------------
# Module developer
# ---------------------------------------------------------------------------


def _rel_path(absolute: str) -> str:
    """Convert an absolute module path to a project-relative POSIX path."""
    return Path(absolute).relative_to(PROJECT_DIR).as_posix()


def _test_file_path(module: str) -> Path:
    """Return the conventional test file path for a module."""
    test_dir = PROJECT_DIR / "tests"
    parts = module.split(".")
    if len(parts) == 1:
        return test_dir / f"{parts[0]}_test.cpp"
    return test_dir / "/".join(parts[:-1]) / f"{parts[-1]}_test.cpp"


def _children_dir(module: str) -> Path:
    """Return the directory where children of *module* would live."""
    source_root = PROJECT_DIR / "src"
    if module == "main":
        return source_root
    return source_root / module.replace(".", "/")


def _module_paths_prompt(agent: ModuleDeveloper) -> str:
    """Callable system prompt that resolves the agent's own file paths."""
    if agent.module == "main":
        source = _rel_path(module_source_path("main"))
        return (
            f"Your module's source file is `{source}` "
            f"(the root module has no header)."
        )
    header = _rel_path(module_header_path(agent.module))
    source = _rel_path(module_source_path(agent.module))
    return (
        f"Your module's header is at `{header}` "
        f"and source is at `{source}`."
    )


class ModuleDeveloper(Developer):
    """Handles the full development lifecycle for a single module."""

    validation_rules = ["build", "test"]

    def __str__(self) -> str:
        return f"developer@{self.module}"

    system_prompts = [
        _module_paths_prompt,
        """\
You are developer@{module}.
You are responsible for all development work on this module: architecture,
API design, implementation, and testing.

Your mandate covers:

- Decomposing your module into sub-modules (via add_module) when warranted

- Writing and maintaining the module's header file: the leading comment block
  describing purpose/responsibilities/dependencies, plus all public API
  declarations (types, function signatures, constants)

- Writing and maintaining the module's source file: the implementation of
  all declared APIs, plus any private utility code

- Writing and maintaining tests for the module's public API

- Delegating sub-module work to child developers via delegate_to_child

Recommended workflow:

1. Read the module's header comment to understand purpose and responsibilities.
   If the module needs sub-modules, create them and delegate their development
   first (in dependency order).

2. Design the API: write declarations in the header. Build to verify.

3. Write tests against the declared API. Build to verify.

4. Implement: fill in function bodies in the source file. Build and run tests.

This sequence is guidance, not a rigid pipeline. Use judgment -- a bug fix
may only need a source-file edit; a trivial leaf module may not need
sub-module decomposition.

Guidelines:

- Prefer flat module hierarchies. About 3-5 sub-modules is the sweet spot.
  Only create sub-modules for distinct concerns likely to take 100+ lines
  of non-trivial code.

- Build frequently to catch compilation errors early. Run tests when you
  believe your implementation is complete -- not after every edit. If a test
  fails and the cause is not immediately clear from the output, finish your
  remaining work rather than debugging; validation runs automatically after
  your task completes and will give you a structured chance to fix failures.

- When delegating to child developers, keep the task description to 1-3
  sentences stating the goal. Child developers can read the codebase
  themselves -- do not echo file contents or prescribe designs.

- Write clean, readable code with comments that explain WHY, not WHAT.

- Test files use the doctest framework. Define
  DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN before including doctest.h.

- You can only write files for your own module. If you need changes in a
  parent or sibling module, call raise_error with a clear explanation so
  your supervisor can address it."""
    ]
    tools = [
        edit_file,
        create_file,
        add_module,
        delegate_to_child,
        dependency_order,
        module_status,
        build,
        run_tests,
    ]

    def _instance_file_access(self) -> list[FileAccessRule]:
        rules: list[FileAccessRule] = []
        if self.module == "main":
            rules.append(FileAccessRule(
                module_source_path(self.module), FileAccessLevel.WRITE,
            ))
        else:
            rules.append(FileAccessRule(
                module_header_path(self.module), FileAccessLevel.WRITE,
            ))
            rules.append(FileAccessRule(
                module_source_path(self.module), FileAccessLevel.WRITE,
            ))
        rules.append(FileAccessRule("tests/**", FileAccessLevel.WRITE))
        return rules

    def context_seed_items(self) -> dict[SeedContent, float]:
        from thorn.core._context_injection import DirectorySeed, FileSeed

        seeds: dict[SeedContent, float] = {}

        if self.module == "main":
            seeds[FileSeed(path=module_source_path("main"))] = 1.0
        else:
            seeds[FileSeed(path=module_header_path(self.module))] = 1.0
            seeds[FileSeed(path=module_source_path(self.module))] = 1.0

        test_path = _test_file_path(self.module)
        if test_path.exists():
            seeds[FileSeed(path=str(test_path))] = 0.5

        if self.module != "main" and "." in self.module:
            parent_module = self.module.rsplit(".", 1)[0]
            seeds[FileSeed(path=module_header_path(parent_module))] = 0.5

        subdir = _children_dir(self.module)
        if subdir.is_dir():
            seeds[DirectorySeed(path=str(subdir))] = 0.5

        return seeds



# ---------------------------------------------------------------------------
# Concierge (injects system prompts into the top-level thorn agent)
# ---------------------------------------------------------------------------


class Concierge(Agent):
    """Workflow-specific guidance for the top-level thorn agent."""

    system_prompts = [
        "This project uses a development workflow where specialized tools "
        "handle all source code changes. When the user requests development "
        "work (implementing features, fixing bugs, refactoring, designing "
        "APIs, adding tests, etc.), delegate the ENTIRE task through the "
        "appropriate development tool. Do not plan, decompose, or design "
        "solutions yourself — state the user's goal concisely and let the "
        "development team determine the approach.",
    ]
