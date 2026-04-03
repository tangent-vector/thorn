# Modular C++ Development Workflow

A hierarchical development workflow for C++ projects using
[thorn](../../../README.md).  A single **developer agent** per module
handles the full development lifecycle (architecture, API design,
implementation, testing), delegating sub-module work to child developers.
Deterministic validation (build, test) is enforced after each delegation.

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

### 2. Run the developer

```
thorn run "coordinate 'implement a stack-based expression evaluator'"
```

The `coordinate` tool creates a `developer@main` agent that will:

1. Inspect the current project state and read `main.cpp`
2. Decompose the project into sub-modules if appropriate
3. Delegate sub-module development to child developers
4. Design the API, write tests, and implement the module's own code
5. Validate (build + test) after completing work

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
        └─> developer@main
              ├─> developer@parser
              │     ├─> developer@parser.lexer
              │     └─> developer@parser.ast
              ├─> developer@evaluator
              └─> developer@repl
```

Each agent is identified as `developer@module`.  A developer handles
all work for its own module (header, source, tests) and delegates
**downward** to child developers for sub-module work.

### The Developer Role

| Capability                | Files written           |
|---------------------------|-------------------------|
| Architecture/decomposition | Header comments, creates sub-modules |
| API design                | Header declarations     |
| Implementation            | Source file             |
| Testing                   | Test files              |
| Delegation                | None (creates child developers) |

Every developer is scoped to a single module.  `developer@parser` can
write `parser.h`, `parser.cpp`, and test files for `parser` -- but not
`parser/lexer.cpp` or `main.cpp`.

If a developer cannot complete its task within its allowed scope (e.g.
it discovers a broken API in a sibling module), it **raises an error**
rather than exceeding its mandate.  The parent developer sees the error
and decides how to proceed.

### Delegation Flow

```
developer@module
  │
  ├─ (reads codebase, designs API, writes header)
  │    → build validation
  │
  ├─ delegate_to_child("parser", task)
  │    └─ developer@parser runs (same pattern recursively)
  │         → build + test validation on completion
  │
  ├─ (writes tests, implements source)
  │    → build + test validation on completion
  │
  └─ final build + test validation by orchestration
```

After each delegation completes, **validation rules** run
deterministically.  If validation fails (e.g. build error), the agent
is given the errors and retries automatically, up to a configurable
limit.

### Validation Rules

Validation checks are registered in `orchestration.py`:

```python
VALIDATION_CHECKS = {
    "build": build,       # from build_tools.py
    "test":  run_tests,   # from build_tools.py
}
```

`ModuleDeveloper` declares `validation_rules = ["build", "test"]`, so
both checks run after each developer completes its task.

The effective rules for any agent are computed as:

    (role_defaults | explicitly_enabled) - explicitly_disabled

Explicit overrides propagate through the delegation chain via
`ContextVar`s.  Each delegation can skip or enable rules:

```python
delegate_to_child("parser", task, skip_validation=["test"])
```

## File Layout

| File                | Purpose                                              |
|---------------------|------------------------------------------------------|
| `build_tools.py`    | CMake configure/build/clean/run tools                |
| `module_tools.py`   | Module tree navigation + `module_status` inspection  |
| `orchestration.py`  | Delegation, validation, `coordinate` entry point     |
| `roles.py`          | Agent class definitions (`ModuleDeveloper`, etc.)    |

Files use relative imports to reference each other (e.g.
`from .orchestration import delegate_to_child` in `roles.py`).

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
  task to a developer agent.  This is the primary entry point for any
  development work.
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

### Developer delegation tools (not available to concierge)

- **`delegate_to_child(child, task, ...)`** -- create a developer for
  a child module to handle sub-module work

## Extending the Workflow

### Adding a validation rule

1. Register the check function in `VALIDATION_CHECKS` in
   `orchestration.py`:

```python
VALIDATION_CHECKS["lint"] = run_linter   # your check function
```

2. Add the rule name to the `validation_rules` list on `ModuleDeveloper`
   (in `roles.py`):

```python
class ModuleDeveloper(Developer):
    validation_rules = ["build", "test", "lint"]
```

Callers can also enable rules dynamically via
`enable_validation=["lint"]`.

## Separation of Concerns

The file organization anticipates future factoring:

- **Build system specifics** (CMake): `build_tools.py`
- **Language conventions** (C++): `module_tools.py` + prompts in `roles.py`
- **Core orchestration**: `orchestration.py` (mostly generic)
- **Role definitions**: `roles.py` (mix of generic and project-specific)

`orchestration.py` deliberately avoids importing role classes directly,
resolving the `ModuleDeveloper` class at runtime through the `Agent`
subclass registry.  This makes the orchestration logic extractable into
`thorn` core without pulling in project-specific code.
