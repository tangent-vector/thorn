"""Agent role definitions for the modular C++ development workflow.

Defines the role hierarchy:

- ``Developer`` (abstract): shared project knowledge and inspection tools
- ``WorkflowRole`` (abstract): base for delegatable roles scoped to a
  single module (Architect, APIDesigner, TestEngineer, Implementer)
- ``Coordinator``: orchestrates work across a module subtree via delegation
- ``Concierge``: injects system prompts into the top-level thorn agent
"""

from __future__ import annotations

from thorn import Agent, read_file, run_shell, write_file
from thorn.tools import list_directory

from .build_tools import build, run_calc
from .module_tools import (
    add_module,
    dependency_order,
    list_all_modules,
    list_submodules,
    module_header_path,
    module_source_path,
    module_status,
)
from .orchestration import (
    delegate_to_child,
    delegate_to_role,
    register_role,
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

_MANDATE_WARNING = (
    "If completing your task is impossible within your allowed scope "
    "(for example, because a dependency has bugs, a required API is "
    "missing, or a test appears to be incorrect), you MUST call "
    "raise_error with a clear explanation rather than attempting work "
    "outside your mandate."
)

# ---------------------------------------------------------------------------
# Abstract bases
# ---------------------------------------------------------------------------


class Developer(Agent, abstract=True):
    """Base for all workflow agents with shared project knowledge."""

    system_prompts = [
        "You are working on a C++ project.",
        _COMMENT_CONVENTION,
        _FILESYSTEM_CONVENTION,
    ]
    tools = [
        read_file, list_directory, list_submodules,
        module_header_path, module_source_path, list_all_modules,
    ]


class WorkflowRole(Developer, abstract=True):
    """Base for delegatable development roles scoped to a single module."""

    system_prompts = [
        "You are ONLY responsible for module `{module}` itself -- not "
        "its parent, not its children. Each module has its own "
        "responsible agent.",
        _MANDATE_WARNING,
    ]


# ---------------------------------------------------------------------------
# Delegatable roles
# ---------------------------------------------------------------------------


class Architect(WorkflowRole):
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


register_role("architect", Architect)


class APIDesigner(WorkflowRole):
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


register_role("api_designer", APIDesigner)


class TestEngineer(WorkflowRole):
    """Writes black-box tests against declared APIs."""

    system_prompts = [
        "You are test_engineer@{module}. You write black-box tests that "
        "exercise the public API declared in the module's header.",
        "Do not modify the module's header or source file. Only create or "
        "modify test files.",
    ]
    tools = [write_file]


register_role("test_engineer", TestEngineer)


class Implementer(WorkflowRole):
    """Implements declared APIs in source files."""

    system_prompts = [
        "You are implementer@{module}. You fill in the implementation in "
        "the module's .cpp source file to satisfy the declarations in its "
        "header.",
        "Do not modify the header file or test files. Build and run tests "
        "to verify your implementation.",
    ]
    tools = [write_file, run_shell, build, run_calc]


register_role("implementer", Implementer)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class Coordinator(Developer):
    """Orchestrates development work across a module subtree via delegation."""

    system_prompts = [
        "You are coordinator@{module}. You coordinate development work "
        "for this module and its subtree by delegating to specialized "
        "roles and child coordinators.",

        "DELEGATION AUTHORITY:\n"
        "- delegate_to_role: Invoke a development role at YOUR module. "
        "Available roles: architect, api_designer, test_engineer, "
        "implementer.\n"
        "- delegate_to_child: Invoke a coordinator at a direct child "
        "module (use list_submodules to discover children).\n"
        "You can ONLY delegate downward. You cannot modify files directly, "
        "invoke roles at other modules, or delegate to your parent or "
        "siblings.",

        "AVAILABLE ROLES:\n"
        "- architect: Decomposes the module into sub-modules and defines "
        "high-level structure. Writes description comments and creates "
        "sub-module files. Does NOT write code.\n"
        "- api_designer: Designs the public API by writing type definitions "
        "and function signatures in the header. Declarations only.\n"
        "- test_engineer: Writes black-box tests against the declared API. "
        "Only creates/modifies test files.\n"
        "- implementer: Fills in function bodies in .cpp source files. "
        "Uses the build tool to verify compilation.",

        "TYPICAL SEQUENCING (guidance, not rigid rules):\n"
        "1. Architecture: delegate_to_role('architect', ...) to define "
        "module structure and sub-modules.\n"
        "2. API design: delegate_to_role('api_designer', ...) to declare "
        "interfaces. Design dependencies before dependents.\n"
        "3. Testing: delegate_to_role('test_engineer', ...) to write "
        "tests against the declared API.\n"
        "4. Implementation: delegate_to_role('implementer', ...) to "
        "write the code. Implement dependencies before dependents.\n"
        "For existing codebases, inspect module state first and skip "
        "steps that are already done. Adapt the sequence to the task.",

        "BEFORE DELEGATING, INSPECT:\n"
        "Use module_status, list_submodules, dependency_order, and "
        "read_file to understand current state. Don't assume the "
        "module needs all phases -- check what exists first.",

        "ERROR HANDLING:\n"
        "If a delegated agent reports an error, you may retry with "
        "adjusted instructions, try a different approach, or delegate "
        "to a different role. If the task truly cannot be completed "
        "within your authority (your module and its descendants), you "
        "MUST call raise_error with a clear explanation so your parent "
        "can decide what to do.",
    ]
    tools = [
        delegate_to_role,
        delegate_to_child,
        dependency_order,
        module_status,
    ]


# ---------------------------------------------------------------------------
# Concierge (injects system prompts into the top-level thorn agent)
# ---------------------------------------------------------------------------


class Concierge(Agent):
    """Workflow-specific guidance for the top-level thorn agent."""

    system_prompts = [
        "This project uses a coordinator-based development workflow. "
        "For any task that involves modifying the project's source code "
        "(implementing features, fixing bugs, refactoring, designing "
        "APIs, adding tests, etc.), you MUST use the `coordinate` tool. "
        "Do not attempt to edit source files or invoke development roles "
        "directly.",
    ]
