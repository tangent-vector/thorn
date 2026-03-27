# Modular C++ Development Workflow

A structured, agent-driven workflow for building C++ projects from scratch
("vibe-coding"). It uses [thorn](../../../README.md) to orchestrate a
pipeline of LLM-powered agents, each scoped to a single module and a single
role, with deterministic Python code handling the sequencing, file layout,
and dependency ordering.

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
    dev_tools.py
```

### 1. Create the seed files

The workflow needs two files to start from:

**`CMakeLists.txt`** — Minimal CMake configuration. The key requirement is
that it globs `src/*.cpp` recursively and sets include directories to `src/`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES src/*.cpp)
add_executable(my_project ${SOURCES})
target_include_directories(my_project PRIVATE src)
```

**`src/main.cpp`** — The root module. This file is the seed for the entire
architecture. Write a detailed leading comment block describing *what* the
application should do, its responsibilities, requirements, and constraints.
The `main()` function body can be empty — it will be filled in during the
implementation phase.

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
//   - Each one may become a module during architecture
//
// Requirements/Constraints:
//   - Language standard, dependency constraints, etc.
//

int main()
{
    return 0;
}
```

The quality of this description directly affects the quality of the
architecture the agents produce. Be specific about what the application
should do, but leave *how* to the agents.

### 2. Architecture

```
thorn run "fully_architect main"
```

The `fully_architect` tool recursively decomposes the project into modules:

1. An **Architect** agent reads `main.cpp`, fleshes out the description,
   and decides what child modules are needed.
2. For each child module, it creates header and source files via the
   `add_module` tool.
3. The process recurses into each new child module.

After this step, inspect the file tree and header files. You should see
module files with populated Purpose/Responsibilities/Dependencies comment
blocks, but no actual code declarations or implementations.

**What to look for:**
- A flat-ish hierarchy (most modules should be leaves with no children)
- 3-5 child modules at most per parent
- Clear, distinct responsibilities for each module
- No code — only structured comments

If the decomposition is too deep or too fine-grained, you can delete the
offending files and re-run `fully_architect` on the parent module.

### 3. API Design

```
thorn run "design_all_apis main"
```

The `design_all_apis` tool walks all modules in bottom-up dependency order
and has an **API Designer** agent write type definitions, class declarations,
and function signatures in each module's header file.

After this step, headers should contain complete API declarations (types,
classes, function prototypes) but no function bodies.

**What to look for:**
- Clean, consistent type hierarchies
- Proper use of namespaces mirroring the module hierarchy
- Forward declarations and includes that reflect actual dependencies
- No implementations — declarations only

### 4. Implementation

```
thorn run "implement_all main"
```

The `implement_all` tool walks all modules in bottom-up dependency order
and has an **Implementer** agent fill in the `.cpp` source files.

The Implementer has access to the build tool and can (and should) build
the project to verify compilation after writing code.

### 5. Build and verify

```
thorn run "build"
```

Or manually:

```
cmake -S . -B build
cmake --build build
```

Run the resulting binary to verify it works as expected.

### 6. (Optional) Testing

```
thorn run "test_all main"
```

The `test_all` tool has a **Test Engineer** agent write black-box tests for
each module. This step is currently best used before implementation (to
define expectations) but the test framework integration (CTest/Catch2) is
still a work in progress.

## Filesystem Conventions

Modules have dot-separated qualified names reflecting their hierarchy:
`parser`, `parser.lexer`, `parser.lexer.token`.

The filesystem layout follows these rules:

| Qualified name       | Header path              | Source path              |
|----------------------|--------------------------|--------------------------|
| `expression`         | `src/expression.h`       | `src/expression.cpp`     |
| `parser`             | `src/parser.h`           | `src/parser.cpp`         |
| `parser.lexer`       | `src/parser/lexer.h`     | `src/parser/lexer.cpp`   |
| `parser.lexer.token` | `src/parser/lexer/token.h` | `src/parser/lexer/token.cpp` |

Key rules:
- **Module code** lives at `parent_dir/name.h` + `parent_dir/name.cpp`
- **Children directory**: if a module has children, they live in
  `parent_dir/name/`
- **Root module** (`main`): `src/main.cpp` only — no header. Its children
  are files directly in `src/`.
- **Include paths** are relative to `src/`: `#include "expression.h"`,
  `#include "parser/lexer.h"`

No files need to move when a leaf module gains or loses children.

## Comment Conventions

All code files use structured leading comments with these sections:

- **Purpose**: 1-2 sentences on what the module does
- **Responsibilities**: Bullet list of this module's responsibilities
- **Dependencies**: Internal project modules and external libraries
- **Requirements/Constraints**: Design constraints (optional)

Do NOT include a "Sub-modules" section. Sub-module structure is determined
by the filesystem, not by comments.

## Roles

| Role            | Responsibility                                          | Can write files? |
|-----------------|---------------------------------------------------------|------------------|
| **Architect**   | Decompose modules, define structure, create sub-modules | Headers only (comments) |
| **API Designer**| Write type definitions and function declarations        | Headers only     |
| **Test Engineer** | Write black-box tests against declared APIs           | Test files only  |
| **Implementer** | Fill in function bodies in `.cpp` files                 | Source files     |
| **Coordinator** | Inspect state, delegate work to other roles             | No               |

Each role is scoped to a single module. An `implementer@parser` agent is
responsible for `parser.cpp` only — not `parser/lexer.cpp`, not `main.cpp`.

## Deterministic Tools Reference

These Python tools are available to agents and can also be called directly
from workflow code:

- **`list_submodules(name)`** — List direct child module names
- **`module_header_path(name)`** — Resolve qualified name to header path
- **`module_source_path(name)`** — Resolve qualified name to source path
- **`add_module(name, parent, description)`** — Create a new child module
- **`list_all_modules(root)`** — Recursively list all modules in subtree
- **`dependency_order(root)`** — Topological sort for bottom-up traversal

## Workflow Tools Reference

These are the high-level orchestration tools available via `thorn run`:

- **`fully_architect <name>`** — Architect a module tree recursively
- **`design_all_apis <root>`** — Design APIs bottom-up in dependency order
- **`test_all <root>`** — Write tests bottom-up in dependency order
- **`implement_all <root>`** — Implement all modules bottom-up

Build tools (from `build_tools.py`):

- **`configure`** — Run CMake configure step
- **`build`** — Build the project (auto-configures if needed)
- **`clean`** — Remove the build directory
- **`run_calc`** — Run the built binary with optional stdin input

## Troubleshooting

**Over-decomposition**: If the Architect creates too many modules or too
deep a hierarchy, the system prompts contain guidance to prefer flat
architectures. You can also edit `dev_tools.py` to tighten the constraints
in the `Architect` role's `system_prompts`.

**Agents modifying wrong files**: Each role's system prompts instruct it
to only modify specific file types. Currently this is enforced by
convention (system prompt instructions), not by the framework. A future
version will add path-based write restrictions per role.

**Build failures after implementation**: The Implementer agent has access
to the `build` tool but may not always use it. See the roadmap section
below for plans to make build verification deterministic.

## Roadmap: Reducing Manual Steps

The current workflow requires the user to invoke four separate commands
(architect, design, implement, build) and manually verify between steps.
Several improvements are planned:

**Single-command pipeline**: A `develop_all` tool that chains the entire
pipeline — architecture through build verification — in a single
invocation. This is straightforward to implement as a `@tool` function
that calls the existing workflow tools in sequence.

**Deterministic build verification**: Currently the Implementer agent
*may* call the build tool, but this is at its discretion. A better design
would make `implement_module` a hybrid Python function that calls a
`@skill` to have the LLM write code, then deterministically runs the
build. On failure, the build errors would be fed back to the agent for
a fix attempt, with a retry limit. This ensures `implement_all` cannot
return success without a clean build.

**Coordinator-driven workflow**: The `Coordinator` role is defined but
does not yet have delegation tools. Once delegation is implemented, a
coordinator agent could inspect the state of the project (which modules
exist, which have APIs, which compile) and decide what work to do next,
replacing the rigid step-by-step pipeline with adaptive decision-making.

**Status inspection**: A `status` tool that reports the current state of
each module (exists? has API declarations? has implementation? compiles?)
would help both human users and coordinator agents understand where the
project stands.
