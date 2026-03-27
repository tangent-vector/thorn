"""Deterministic module management tools for hierarchical C++ projects.

Pure Python functions for navigating and manipulating the module tree.
No LLM calls -- these provide the structural backbone that workflow
agents build on top of.
"""

from __future__ import annotations

import re
from pathlib import Path

from thorn import tool

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_DIR / "src"


def qualify(parent: str, child: str) -> str:
    """Build a qualified module name from a parent and unqualified child name.

    The root module ``"main"`` is special: its children use simple names
    (e.g. ``"parser"``, not ``"main.parser"``).
    """
    return child if parent == "main" else f"{parent}.{child}"


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
    qualified = qualify(parent, name)

    header = Path(module_header_path(qualified))
    source = Path(module_source_path(qualified))

    if header.exists():
        raise FileExistsError(f"Header already exists: {header}")
    if source.exists():
        raise FileExistsError(f"Source already exists: {source}")

    header.parent.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)

    include_path = header.relative_to(SOURCE_ROOT).as_posix()
    guard = f"{qualified.upper().replace('.', '_')}_H"

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
        result.extend(list_all_modules(qualify(root, child)))
    return result


_INCLUDE_RE = re.compile(r'#include\s+"([^"]+)"')


@tool
def dependency_order(root: str) -> list[str]:
    """Return modules in bottom-up dependency order (dependencies first).

    Parses #include "..." directives from header files to build a
    dependency graph among modules in the tree. Uses Kahn's algorithm
    for topological sorting.

    Raises ValueError if cyclic dependencies are detected.
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
        raise ValueError(
            f"Cyclic dependencies detected among modules: {remaining}"
        )

    return result
