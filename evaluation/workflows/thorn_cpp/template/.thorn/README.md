# Modular C++ Development Workflow

A coordinator-based development workflow for C++ projects using
[thorn](../../../README.md).  Instead of rigid phase-by-phase tools,
a hierarchy of **coordinator agents** autonomously decompose tasks and
delegate to specialized **roles**, with deterministic validation
(build, test) enforced after each delegation.

## Prerequisites

- **thorn** installed (`pip install -e .` from the repo root)
- An **LLM provider** configured (e.g. `OPENAI_API_KEY` in a `.env` file
  at the project root or in your environment)
- **CMake** (>= 3.20) and a C++ compiler (MSVC, GCC, or Clang)

## Quick Start

Starting from a directory that contains only this `.thorn/` directory:

```
my_project/
  .thorn/
    build_tools.py
    module_tools.py
    orchestration.py
    roles.py
```

### 1. Create the seed files

The workflow needs two files to start from:

**`CMakeLists.txt`** -- minimal CMake configuration:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES src/*.cpp)
add_executable(my_project ${SOURCES})
target_include_directories(my_project PRIVATE src)
```

**`src/main.cpp`** -- the root module.  Write a detailed leading comment
describing what the application should do.  The `main()` body can be
empty -- it will be filled in during implementation:

```cpp
// ============================================================================
// my_project — brief title
// ============================================================================
//
// Purpose:
//   What the application does, in 1-3 sentences.
//
// Responsibilities:
//   - Bullet list of top-level responsibilities
//
// Requirements/Constraints:
//   - Language standard, dependency constraints, etc.
//

int main()
{
    return 0;
}
```

### 2. Run the coordinator

```
thorn run "coordinate 'implement a stack-based expression evaluator'"
```

The `coordinate` tool creates a `coordinator@main` agent that will:

1. Inspect the current project state
2. Delegate to `architect@main` to decompose into modules
3. Delegate to child coordinators for each sub-module
4. Within each module, delegate to API designers, test engineers, and
   implementers as needed
5. Validate (build + test) after implementation, retrying on failure

You can also target a specific module:

```
thorn run "coordinate 'add error recovery' --module parser"
```

Or use `thorn chat` for interactive sessions:

```
thorn chat
you> coordinate "fix the division-by-zero bug in the evaluator"
```

## Architecture

### Agent Hierarchy

```
concierge (thorn run / thorn chat)
  └─> coordinate(task)
        └─> coordinator@main
              ├─> architect@main
              ├─> api_designer@main
              ├─> implementer@main
              ├─> coordinator@parser
              │     ├─> architect@parser
              │     ├─> api_designer@parser
              │     ├─> coordinator@parser.lexer
              │     │     └─> ...
              │     └─> ...
              └─> ...
```

Each agent is identified as `role@module`.  Coordinators can only
delegate **downward**: to roles at their own module, or to coordinators
at child modules.

### Roles

| Role              | Responsibility                                       | Can write            |
|-------------------|------------------------------------------------------|----------------------|
| **Coordinator**   | Inspect state, decompose tasks, delegate to roles/children | Nothing (read-only)  |
| **Architect**     | Decompose modules, define structure, create sub-modules | Header comments only |
| **API Designer**  | Write type definitions and function declarations     | Headers only         |
| **Test Engineer** | Write black-box tests against declared APIs          | Test files only      |
| **Implementer**   | Fill in function bodies in `.cpp` files              | Source files         |

Every role is scoped to a single module.  `implementer@parser` is
responsible for `parser.cpp` only -- not `parser/lexer.cpp`, not
`main.cpp`.

If a role cannot complete its task within its allowed scope (e.g. an
implementer finds a broken test, or an API designer discovers a missing
dependency), it **raises an error** rather than exceeding its mandate.
The coordinator sees the error and decides how to proceed.

### Delegation Flow

```
coordinator@module
  │
  ├─ delegate_to_role("architect", task)
  │    └─ architect@module runs → validation → retry on fail
  │
  ├─ delegate_to_role("api_designer", task)
  │    └─ api_designer@module runs → validation → retry on fail
  │
  ├─ delegate_to_child("parser", task)
  │    └─ coordinator@parser runs (same pattern recursively)
  │
  └─ delegate_to_role("implementer", task)
       └─ implementer@module runs → validation → retry on fail
```

After each delegation, **validation rules** run deterministically.
If validation fails (e.g. build error), the sub-agent is given the
errors and retries automatically, up to a configurable limit.

### Validation Rules

Validation checks are registered in `orchestration.py`:

```python
VALIDATION_CHECKS = {
    "build": build,       # from build_tools.py
    "test":  run_tests,   # from build_tools.py
}
```

Each agent role declares which validation rules apply to it via a
`validation_rules` class attribute (accumulated through the MRO, like
`system_prompts` and `tools`).  For example, the `Implementer` and
`Coordinator` roles declare `validation_rules = ["build", "test"]`,
while `Architect`, `APIDesigner`, and `TestEngineer` have no validation
rules by default.

The effective rules for any agent are computed as:

    (role_defaults | explicitly_enabled) - explicitly_disabled

Explicit overrides propagate through the delegation chain via
`ContextVar`s.  Each delegation can skip or enable rules:

```
delegate_to_role("architect", task, skip_validation=["build"])
```

## File Layout

| File                | Purpose                                              |
|---------------------|------------------------------------------------------|
| `build_tools.py`    | CMake configure/build/clean/run tools                |
| `module_tools.py`   | Module tree navigation + `module_status` inspection  |
| `orchestration.py`  | Delegation, validation, `coordinate` entry point     |
| `roles.py`          | Role definitions (all Agent subclasses)              |

Files use relative imports to reference each other (e.g.
`from .orchestration import delegate_to_role` in `roles.py`).

## Module Conventions

Modules have dot-separated qualified names reflecting their hierarchy:

| Qualified name       | Header path                | Source path                |
|----------------------|----------------------------|----------------------------|
| `expression`         | `src/expression.h`         | `src/expression.cpp`       |
| `parser`             | `src/parser.h`             | `src/parser.cpp`           |
| `parser.lexer`       | `src/parser/lexer.h`       | `src/parser/lexer.cpp`     |

Key rules:
- **Module code** lives at `parent_dir/name.h` + `parent_dir/name.cpp`
- **Children directory**: if a module has children, they live in
  `parent_dir/name/`
- **Root module** (`main`): `src/main.cpp` only -- no header
- **Include paths** are relative to `src/`

## Tool Reference

### Concierge tools (available via `thorn run` / `thorn chat`)

- **`coordinate(task, module="main", ...)`** -- delegate a development
  task to the coordinator hierarchy.  This is the primary entry point
  for any development work.
- **`build`** -- build the project via CMake
- **`configure`** -- run CMake configure step
- **`clean`** -- remove the build directory
- **`run_calc`** -- run the built binary with optional stdin
- **`module_status(name, query="")`** -- inspect a module's state
- **`list_submodules(name)`** -- list direct children
- **`dependency_order(root)`** -- topological sort of modules
- **`add_module(name, parent, description)`** -- create a new module
- **`module_header_path(name)`** / **`module_source_path(name)`** --
  resolve qualified names to file paths

### Coordinator delegation tools (not available to concierge)

- **`delegate_to_role(role, task, ...)`** -- invoke a role at the
  coordinator's own module
- **`delegate_to_child(child, task, ...)`** -- invoke a coordinator at
  a child module

## Extending the Workflow

### Adding a new role

Define a new `WorkflowRole` subclass in `roles.py` and register it:

```python
class SecurityAuditor(WorkflowRole):
    system_prompts = [
        "You are security_auditor@{module}. Review the module's code "
        "for security vulnerabilities.",
    ]
    tools = [write_file]

register_role("security_auditor", SecurityAuditor)
```

The coordinator will automatically see the new role as a delegation
target.

### Adding a validation rule

1. Register the check function in `VALIDATION_CHECKS` in
   `orchestration.py`:

```python
VALIDATION_CHECKS["lint"] = run_linter   # your check function
```

2. Add the rule name to the `validation_rules` list on the agent roles
   that should be validated by it (in `roles.py`):

```python
class Implementer(WorkflowRole):
    validation_rules = ["build", "test", "lint"]
```

Only roles that list the rule will be validated by it.  Callers can
also enable it dynamically via `enable_validation=["lint"]`.

## Separation of Concerns

The file organization anticipates future factoring:

- **Build system specifics** (CMake): `build_tools.py`
- **Language conventions** (C++): `module_tools.py` + prompts in `roles.py`
- **Core orchestration**: `orchestration.py` (mostly generic)
- **Role definitions**: `roles.py` (mix of generic and project-specific)

`orchestration.py` deliberately avoids importing role classes directly,
resolving everything through the role registry and `Agent` subclass
registry.  This makes the orchestration logic extractable into `thorn`
core without pulling in project-specific code.
