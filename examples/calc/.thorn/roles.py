"""Agent role definitions for the modular C++ development workflow.

Defines the role hierarchy:

- ``Developer`` (abstract): shared project knowledge and inspection tools
- ``WorkflowRole`` (abstract): base for delegatable roles scoped to a
  single module (Architect, APIDesigner, TestEngineer, Implementer)
- ``Coordinator``: orchestrates work across a module subtree via delegation
- ``Concierge``: injects system prompts into the top-level thorn agent
"""

from __future__ import annotations

from thorn import Agent, FileAccessLevel, FileAccessRule, read_file, write_file
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

_SUMMARY_GUIDANCE = (
    "When reporting what you did, be concise: use a short bulleted list "
    "of actions taken. Do NOT generate ASCII-art trees, directory "
    "listings, code fences with file contents, or any other repetitive "
    "structured output in your summary. Just state what you created, "
    "modified, or verified."
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
        _SUMMARY_GUIDANCE,
    ]
    tools = [
        read_file, list_directory, list_submodules,
        module_header_path, module_source_path, list_all_modules,
    ]
    file_access = [
        FileAccessRule("**", FileAccessLevel.READ),
        FileAccessRule(".thorn", FileAccessLevel.HIDDEN),
        FileAccessRule(".thorn/**", FileAccessLevel.HIDDEN),
    ]


class WorkflowRole(Developer, abstract=True):
    """Base for delegatable development roles scoped to a single module."""

    def __str__(self) -> str:
        return f"{type(self).__name__.lower()}@{self.module}"

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
        "into sub-modules and define the high-level structure.\n\n"
        "FILE ACCESS: You have write access ONLY to the module's header "
        "file (or main.cpp for the root module). All other files are "
        "read-only. Use the add_module tool to create new child modules.\n\n"
        "ALLOWED ACTIONS — you may ONLY do these two things:\n"
        "1. Update the Purpose/Responsibilities/Dependencies COMMENT BLOCK "
        "in the module's header file (or main.cpp for the root module). "
        "Do NOT add or modify any code outside the comment block — no "
        "#include directives, no declarations, no namespaces, no code.\n"
        "2. Call add_module to create new child modules. The tool creates "
        "properly formatted scaffold files; do NOT write to those files "
        "yourself.\n\n"
        "NEVER write code. NEVER write #include directives. NEVER write "
        "declarations, definitions, type definitions, or function signatures. "
        "Those are the api_designer's and implementer's jobs.\n\n"
        "NEVER modify .cpp source files (except main.cpp comments for the "
        "root module). NEVER create files via write_file — use add_module.\n\n"
        "If the module's header already has good description comments and "
        "doesn't need sub-modules, say so and you are done.",

        "DESIGN PRINCIPLES:\n"
        "- Prefer FLAT architectures. Most modules should be LEAF modules "
        "with no sub-modules.\n"
        "- Only create a sub-module when it represents a clearly distinct "
        "concern (50+ lines of non-trivial code on its own).\n"
        "- A module with 3-6 related responsibilities should stay as a "
        "single leaf module.\n"
        "- Maximum 3-5 sub-modules per parent. Hierarchy rarely exceeds "
        "2 levels deep.\n"
        "- Use clear, descriptive names (e.g., 'expression' not 'expr', "
        "'evaluator' not 'eval'). Avoid abbreviations.",
    ]
    tools = [write_file, add_module]

    def _instance_file_access(self) -> list[FileAccessRule]:
        header = module_header_path(self.module)
        source = module_source_path(self.module)
        rules = [FileAccessRule(header, FileAccessLevel.WRITE)]
        if self.module == "main":
            rules.append(FileAccessRule(source, FileAccessLevel.WRITE))
        return rules


register_role("architect", Architect)


class APIDesigner(WorkflowRole):
    """Designs public API declarations for a module."""

    system_prompts = [
        "You are api_designer@{module}. You design the public API by "
        "writing declarations in the module's HEADER file ONLY.\n\n"
        "FILE ACCESS: You have write access ONLY to the module's header "
        "file. All other files are read-only.\n\n"
        "ALLOWED: Write the complete header file including #include "
        "directives, namespace blocks, type definitions, class definitions, "
        "and function signatures (declarations without bodies).\n\n"
        "FORBIDDEN: Do NOT write function bodies or implementations. "
        "Do NOT modify any .cpp source file. Do NOT modify any other "
        "module's files. You may only write to this module's header.\n\n"
        "Read dependency module headers first to understand available types.",
    ]
    tools = [write_file]

    def _instance_file_access(self) -> list[FileAccessRule]:
        header = module_header_path(self.module)
        return [FileAccessRule(header, FileAccessLevel.WRITE)]


register_role("api_designer", APIDesigner)


class TestEngineer(WorkflowRole):
    """Writes black-box tests against declared APIs."""

    system_prompts = [
        "You are test_engineer@{module}. You write black-box tests that "
        "exercise the public API declared in the module's header.\n\n"
        "FILE ACCESS: You have write access ONLY to test files for this "
        "module. All other files are read-only.\n\n"
        "FORBIDDEN: Do NOT modify the module's header or source file. "
        "Only create or modify test files.",
    ]
    tools = [write_file]

    def _instance_file_access(self) -> list[FileAccessRule]:
        # Grant write to test files in the module's directory
        # For now, grant write to a tests/ directory pattern
        return [FileAccessRule("tests/**", FileAccessLevel.WRITE)]


register_role("test_engineer", TestEngineer)


class Implementer(WorkflowRole):
    """Implements declared APIs in source files."""

    system_prompts = [
        "You are implementer@{module}. You write the implementation in "
        "the module's .cpp SOURCE file to satisfy the declarations in "
        "its header.\n\n"
        "FILE ACCESS: You have write access ONLY to the module's .cpp "
        "source file. All other files are read-only.\n\n"
        "ALLOWED: Write to this module's .cpp source file ONLY. "
        "Use the build tool to verify compilation.\n\n"
        "FORBIDDEN: Do NOT modify the header file. Do NOT modify any "
        "other module's files. Do NOT modify main.cpp (unless you ARE "
        "implementer@main). Do NOT modify test files. Do NOT create "
        "new files.",
    ]
    tools = [write_file, build, run_calc]

    def _instance_file_access(self) -> list[FileAccessRule]:
        source = module_source_path(self.module)
        return [FileAccessRule(source, FileAccessLevel.WRITE)]


register_role("implementer", Implementer)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class Coordinator(Developer):
    """Orchestrates development work across a module subtree via delegation."""

    def __str__(self) -> str:
        return f"coordinator@{self.module}"

    system_prompts = [
        "You are coordinator@{module}. You coordinate development work "
        "for this module and its subtree by delegating to specialized "
        "roles and child coordinators.\n\n"
        "FILE ACCESS: You have read-only access to all project files. "
        "Use delegation tools to make changes.",

        "DELEGATION AUTHORITY:\n"
        "- delegate_to_role(role, task): Invoke a role at YOUR module.\n"
        "- delegate_to_child(child, task): Invoke a coordinator for a "
        "direct child module.\n"
        "You can ONLY delegate downward. You cannot modify files "
        "directly or delegate to your parent or siblings.",

        "AVAILABLE ROLES:\n"
        "- architect: Decomposes the module into sub-modules. Updates "
        "description comments and creates sub-module scaffolds via "
        "add_module. Does NOT write any code.\n"
        "- api_designer: Writes the full header file with type "
        "definitions and function signatures. Declarations only.\n"
        "- implementer: Fills in function bodies in the .cpp source "
        "file. Builds to verify.\n"
        "- test_engineer: Writes tests. Only use when a test framework "
        "is configured; skip otherwise.",

        "WORKFLOW FOR BUILDING A MODULE FROM SCRATCH:\n"
        "1. delegate_to_role('architect', ...) — define structure, "
        "create any sub-modules.\n"
        "2. After architecture, check list_submodules('{module}'). "
        "For each child, use delegate_to_child to have it fully "
        "developed (API + implementation). Process dependencies FIRST.\n"
        "3. delegate_to_role('api_designer', ...) — declare this "
        "module's own API in its header.\n"
        "4. delegate_to_role('implementer', ...) — implement this "
        "module's code.\n"
        "Skip steps that are already done. For existing code, inspect "
        "state first and adapt.",

        "SPECIAL CASE — ROOT MODULE (main):\n"
        "The root module 'main' has NO header file; it is just main.cpp. "
        "Therefore api_designer does NOT apply to 'main'. For the root "
        "module: architect (to create child modules) → develop children "
        "via delegate_to_child → implement main.cpp via implementer.",

        "BEFORE DELEGATING, INSPECT:\n"
        "Use module_status and list_submodules to understand current "
        "state. Don't blindly run all phases.",

        "ERROR HANDLING:\n"
        "If a delegate reports an error, you may retry or try a "
        "different approach. If the task cannot be completed within "
        "your authority, call raise_error with a clear explanation.",
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
