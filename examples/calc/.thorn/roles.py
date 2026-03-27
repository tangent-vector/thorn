"""Agent role definitions for the modular C++ development workflow.

Each role inherits from Developer and defines the system prompts and
tools appropriate for its scope of responsibility.
"""

from __future__ import annotations

from thorn import Agent, read_file, run_shell, write_file
from thorn.tools import list_directory

from .build_tools import build, run_calc
from .module_tools import (
    add_module,
    list_all_modules,
    list_submodules,
    module_header_path,
    module_source_path,
)

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
# Role definitions
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
    tools = [write_file, run_shell, build, run_calc]
