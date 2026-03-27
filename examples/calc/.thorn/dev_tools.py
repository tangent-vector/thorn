"""Development workflow tools and roles for the calc project.

Combines deterministic filesystem-based module management, agent role
definitions, and structured workflow functions.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

from thorn import Agent, read_file, run_shell, skill, tool, write_file
from thorn._tools import list_directory

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_DIR / "src"

# ---------------------------------------------------------------------------
# Import build tools from sibling file (private names to avoid re-discovery)
# ---------------------------------------------------------------------------


def _load_sibling(name: str) -> Any:
    path = Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_thorn_sibling_.{name}", path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sibling module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_build_mod = _load_sibling("build_tools")
_build = _build_mod.build
_run_calc = _build_mod.run_calc

# ---------------------------------------------------------------------------
# Deterministic module tools (pure Python, no LLM calls)
# ---------------------------------------------------------------------------


@tool
def list_submodules(name: str) -> list[str]:
    """List the direct child module names of the given module.

    Returns simple (unqualified) names. For example, list_submodules("parser")
    might return ["lexer", "ast_node"]. Returns [] for leaf modules.
    """
    if name == "main":
        children_dir = SOURCE_ROOT
    else:
        children_dir = SOURCE_ROOT / name.replace(".", "/")

    if not children_dir.is_dir():
        return []
    return sorted(p.stem for p in children_dir.glob("*.h"))


@tool
def module_header_path(name: str) -> str:
    """Return the filesystem path to a module's header file.

    Example: module_header_path("parser.lexer") returns the path to
    src/parser/lexer.h.
    """
    parts = name.split(".")
    dir_path = SOURCE_ROOT
    for p in parts[:-1]:
        dir_path = dir_path / p
    return str(dir_path / (parts[-1] + ".h"))


@tool
def module_source_path(name: str) -> str:
    """Return the filesystem path to a module's source file.

    Example: module_source_path("parser.lexer") returns the path to
    src/parser/lexer.cpp.
    """
    parts = name.split(".")
    dir_path = SOURCE_ROOT
    for p in parts[:-1]:
        dir_path = dir_path / p
    return str(dir_path / (parts[-1] + ".cpp"))


@tool
def add_module(name: str, parent: str, description: str) -> str:
    """Create a new module (header + source) as a child of the given parent.

    Creates the files at the correct filesystem location per the project's
    module hierarchy conventions. The header gets an initial comment block
    with the Purpose section; the source just #includes the header.

    Returns the qualified name of the new module.
    """
    qualified = f"{parent}.{name}" if parent != "main" else name

    header = Path(module_header_path(qualified))
    source = Path(module_source_path(qualified))

    if header.exists():
        raise FileExistsError(f"Header already exists: {header}")
    if source.exists():
        raise FileExistsError(f"Source already exists: {source}")

    header.parent.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)

    include_path = header.relative_to(SOURCE_ROOT).as_posix()
    guard = f"CALC_{qualified.upper().replace('.', '_')}_H"

    header.write_text(
        f"""\
#ifndef {guard}
#define {guard}

// ============================================================================
// Module: {qualified}
// ============================================================================
//
// Purpose:
//   {description}
//
// Responsibilities:
//   (to be defined by architect)
//
// Dependencies:
//   (to be defined by architect)
//

#endif // {guard}
""",
        encoding="utf-8",
    )

    source.write_text(f'#include "{include_path}"\n', encoding="utf-8")

    return qualified


@tool
def list_all_modules(root: str) -> list[str]:
    """Recursively list all module qualified names in the subtree.

    Includes the root itself. Uses filesystem traversal, not LLM calls.
    """
    result = [root]
    for child in list_submodules(root):
        qualified = f"{root}.{child}" if root != "main" else child
        result.extend(list_all_modules(qualified))
    return result


_INCLUDE_RE = re.compile(r'#include\s+"([^"]+)"')


@tool
def dependency_order(root: str) -> list[str]:
    """Return modules in bottom-up dependency order (dependencies first).

    Parses #include "..." directives from header files to build a
    dependency graph among modules in the tree. Uses Kahn's algorithm
    for topological sorting.
    """
    all_modules = list_all_modules(root)
    module_set = set(all_modules)

    deps: dict[str, set[str]] = {m: set() for m in all_modules}

    for mod in all_modules:
        if mod == "main":
            file_path = SOURCE_ROOT / "main.cpp"
        else:
            file_path = Path(module_header_path(mod))

        if not file_path.exists():
            continue

        content = file_path.read_text(encoding="utf-8")
        for match in _INCLUDE_RE.finditer(content):
            inc = match.group(1)
            if inc.endswith(".h"):
                inc = inc[:-2]
            qualified = inc.replace("/", ".").replace("\\", ".")
            if qualified in module_set and qualified != mod:
                deps[mod].add(qualified)

    # Kahn's algorithm
    in_degree = {m: len(deps[m]) for m in all_modules}
    dependents: dict[str, list[str]] = {m: [] for m in all_modules}
    for m, dep_set in deps.items():
        for d in dep_set:
            dependents[d].append(m)

    queue = sorted(m for m in all_modules if in_degree[m] == 0)
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dep in sorted(dependents[node]):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
                queue.sort()

    if len(result) < len(all_modules):
        remaining = sorted(m for m in all_modules if m not in set(result))
        result.extend(remaining)

    return result


# ---------------------------------------------------------------------------
# Shared system prompt fragments
# ---------------------------------------------------------------------------

_COMMENT_CONVENTION = """\
Code files use structured leading comments (C++ style) with these sections:
- Purpose: 1-2 sentences on what the module does
- Responsibilities: Bullet list of this module's responsibilities
- Dependencies: Internal project modules and external libraries
- Requirements/Constraints: Design constraints (optional)

Do NOT include a "Sub-modules" section in comments. Sub-module structure \
is determined by the filesystem (the presence of a name/ directory), not \
by comments."""

_FILESYSTEM_CONVENTION = """\
This project uses a hierarchical module layout under src/:

- Module code: parent_dir/name.h + parent_dir/name.cpp
- Children directory: if module has children, they live in parent_dir/name/
- Root module (main): src/main.cpp is the entry point. \
It has NO header file (main.h does NOT exist). \
Its children are .h/.cpp files directly in src/.
- Include paths are relative to src/. \
Examples: #include "expression.h", #include "parser/lexer.h"
- Namespaces should mirror the module hierarchy (e.g., namespace parser::lexer)

Qualified names use dots: parser.lexer means the file at src/parser/lexer.h. \
The root module "main" is special -- its children use simple names \
(e.g., "parser", not "main.parser"). \
Use the module_header_path and module_source_path tools to resolve paths \
rather than guessing."""

# ---------------------------------------------------------------------------
# Role definitions (project-agnostic names)
# ---------------------------------------------------------------------------


class Developer(Agent):
    """Base for modular development workflow agents."""

    system_prompts = [
        "You are working on module `{module}` of a C++ project.",
        "You are ONLY responsible for `{module}` itself -- not its parent, "
        "not its children. Each module has its own responsible agent.",
        _COMMENT_CONVENTION,
        _FILESYSTEM_CONVENTION,
    ]
    tools = [
        read_file, list_directory, list_submodules,
        module_header_path, module_source_path, list_all_modules,
    ]


class Architect(Developer):
    """Decomposes modules into sub-modules, defines structure."""

    system_prompts = [
        "You are architect@{module}. Your job is to decompose this module "
        "into sub-modules and define the high-level structure.",

        "You write/update description comments in header files and create "
        "sub-module files via the add_module tool. You do NOT write any code "
        "(no declarations, no definitions, no implementations).",

        "IMPORTANT DESIGN PRINCIPLES:\n"
        "- Prefer FLAT architectures. Most modules should be LEAF modules "
        "with no sub-modules.\n"
        "- Only create a sub-module when it represents a clearly distinct "
        "concern that would be large or complex on its own (50+ lines of "
        "non-trivial code).\n"
        "- A module with 3-6 closely related responsibilities should stay "
        "as a single leaf module, NOT be split further.\n"
        "- Aim for a MAXIMUM of 3-5 sub-modules per parent. If you feel "
        "you need more, reconsider whether some responsibilities should be "
        "merged.\n"
        "- The hierarchy should rarely exceed 2 levels deep (parent -> "
        "child). Avoid grandchild modules unless truly warranted.",

        "If this module already has well-written description comments and you "
        "do not see a need for sub-modules, that is a valid outcome. In fact, "
        "most modules should NOT have sub-modules.",
    ]
    tools = [write_file, add_module]


class APIDesigner(Developer):
    """Designs public API declarations for a module."""

    system_prompts = [
        "You are api_designer@{module}. You design the public API by writing "
        "declarations (types, function signatures, class definitions) in the "
        "module's header file.",
        "Write declarations ONLY. Do not write function bodies or "
        "implementations. Read sibling and dependency module headers to "
        "understand the types available.",
    ]
    tools = [write_file]


class TestEngineer(Developer):
    """Writes black-box tests against declared APIs."""

    system_prompts = [
        "You are test_engineer@{module}. You write black-box tests that "
        "exercise the public API declared in the module's header.",
        "Do not modify the module's header or source file. Only create or "
        "modify test files.",
    ]
    tools = [write_file]


class Implementer(Developer):
    """Implements declared APIs in source files."""

    system_prompts = [
        "You are implementer@{module}. You fill in the implementation in "
        "the module's .cpp source file to satisfy the declarations in its "
        "header.",
        "Do not modify the header file or test files. Build and run tests "
        "to verify your implementation.",
    ]
    tools = [write_file, run_shell, _build, _run_calc]


class Coordinator(Developer):
    """Local authority that inspects state and delegates work."""

    system_prompts = [
        "You are coordinator@{module}. You inspect the current state of "
        "your module and its children, decide what work needs to be done, "
        "and delegate to the appropriate role-specific agent.",
        "You do NOT write code or modify files directly.",
    ]
    tools = []


# ---------------------------------------------------------------------------
# Workflow functions
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


@tool
async def fully_architect(name: str) -> None:
    """Architect a module and recursively architect all children."""
    await _architect_module(name)
    for child in list_submodules(name):
        qualified = f"{name}.{child}" if name != "main" else child
        await fully_architect(qualified)


@skill(role=APIDesigner)
async def _design_module_api(module: str) -> None:
    """Design the public API for module `{module}`. Write type definitions
    and function declarations in the header file. Read dependency headers
    to understand available types. Write declarations only -- no
    implementations."""


@tool
async def design_all_apis(root: str) -> None:
    """Design APIs for all modules bottom-up in dependency order."""
    for name in dependency_order(root):
        if name == "main":
            continue
        await _design_module_api(name)


@skill(role=TestEngineer)
async def _write_module_tests(module: str) -> None:
    """Write black-box tests for module `{module}`. Create test files that
    exercise the public API declared in the module's header. Focus on
    correctness and edge cases."""


@tool
async def test_all(root: str) -> None:
    """Write tests for all modules bottom-up in dependency order."""
    for name in dependency_order(root):
        if name == "main":
            continue
        await _write_module_tests(name)


@skill(role=Implementer)
async def _implement_module(module: str) -> None:
    """Implement the declared API for module `{module}` in its .cpp source
    file. Fill in function bodies to satisfy the declarations in the header.
    Build the project after making changes to verify compilation."""


@tool
async def implement_all(root: str) -> None:
    """Implement all modules bottom-up in dependency order."""
    for name in dependency_order(root):
        await _implement_module(name)
