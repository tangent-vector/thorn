"""High-level workflow orchestration for the modular C++ development pipeline.

Each workflow tool chains deterministic traversal with LLM-driven skills
to automate a phase of development (architecture, API design, testing,
implementation) across the module tree.
"""

from __future__ import annotations

from thorn import skill, tool

from .module_tools import dependency_order, list_submodules, qualify
from .roles import APIDesigner, Architect, Implementer, TestEngineer

# ---------------------------------------------------------------------------
# Skills (single-module LLM operations)
# ---------------------------------------------------------------------------


@skill(role=Architect)
async def _architect_module(module: str) -> None:
    """Define architecture for module `{module}`.

    Steps:
    1. Use module_header_path / module_source_path to find the file paths,
       then read the existing files for this module.
    2. Flesh out the Purpose, Responsibilities, and Dependencies sections
       in the header's leading comments (use write_file to update).
    3. Decide whether sub-modules are needed. Most modules should be leaf
       modules with NO sub-modules. Only create sub-modules for clearly
       distinct, large concerns.
    4. If sub-modules are warranted, create them with add_module (max 3-5).
    5. Call return_result when done.

    Do not write code -- only descriptions and structure."""


@skill(role=APIDesigner)
async def _design_module_api(module: str) -> None:
    """Design the public API for module `{module}`.

    Steps:
    1. Use module_header_path to find the header file, then read it.
    2. Read the headers of all modules listed in the Dependencies section
       to understand the types and interfaces available to you.
    3. Read headers of sibling modules (use list_submodules on the parent)
       to ensure consistency across the project.
    4. Write type definitions, class declarations, and function signatures
       in the header file. Preserve the leading comment block and include
       guard. Add #include directives for any dependency headers you use.
    5. Write declarations ONLY -- no function bodies, no implementations.
       Method bodies should be omitted or declared as just a signature.
    6. Call return_result when done.

    Aim for a clean, minimal API. Prefer free functions over classes when
    there is no state to manage. Use the project's namespace conventions
    (namespace matching the module hierarchy)."""


@skill(role=TestEngineer)
async def _write_module_tests(module: str) -> None:
    """Write black-box tests for module `{module}`.

    Steps:
    1. Use module_header_path to find and read the module's header file.
       Understand the public API (types, functions, classes) that you will
       be testing.
    2. Read the module's source file (module_source_path) if it exists, but
       treat the implementation as opaque -- test against the declared API.
    3. Create a test source file at the appropriate location. Use a simple
       test harness (assert-based with a main function) unless a test
       framework is already configured.
    4. Write tests covering:
       - Normal/expected inputs (happy path)
       - Edge cases (empty input, zero, boundary values)
       - Error conditions (invalid input, if the API defines error behavior)
    5. Do NOT modify the module's header or source file.
    6. Call return_result when done."""


@skill(role=Implementer)
async def _implement_module(module: str) -> None:
    """Implement the declared API for module `{module}`.

    Steps:
    1. Use module_header_path to read the header file. Understand every
       type, class, and function you need to implement.
    2. Use module_source_path to find the source file, then read it.
    3. Read the headers of dependency modules (listed in the Dependencies
       comment section) to understand the interfaces you can call.
    4. Write the complete implementation in the .cpp source file. Include
       all necessary headers. Implement every function and method declared
       in the header.
    5. Do NOT modify the header file.
    6. Use the build tool to compile and verify your implementation. If the
       build fails, read the error output and fix the issues. Repeat until
       the build succeeds.
    7. Call return_result when done."""


# ---------------------------------------------------------------------------
# Workflow tools (multi-module orchestration)
# ---------------------------------------------------------------------------


@tool
async def fully_architect(name: str) -> None:
    """Architect a module and recursively architect all children."""
    await _architect_module(name)
    for child in list_submodules(name):
        await fully_architect(qualify(name, child))
    dependency_order(name)


@tool
async def design_all_apis(root: str) -> None:
    """Design APIs for all modules bottom-up in dependency order."""
    for name in dependency_order(root):
        if name == "main":
            continue
        await _design_module_api(name)


@tool
async def test_all(root: str) -> None:
    """Write tests for all modules bottom-up in dependency order."""
    for name in dependency_order(root):
        if name == "main":
            continue
        await _write_module_tests(name)


@tool
async def implement_all(root: str) -> None:
    """Implement all modules bottom-up in dependency order."""
    for name in dependency_order(root):
        await _implement_module(name)
