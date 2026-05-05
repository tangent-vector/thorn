---
name: thorn-development-loop
description: Develop Thorn changes using the project's required uv workflow, CLI/gateway priority model, testing expectations, and API compatibility policy. Use when coding, testing, reviewing, or planning changes in the Thorn repository.
---

# Thorn Development Loop

Use this skill for work in the Thorn repository. It complements the root `AGENTS.md`; if they conflict, follow `AGENTS.md`.

## Project Priorities

Thorn is an end-user application. Prioritize behavior in this order:

1. `thorn` CLI commands such as `thorn run`, `thorn chat`, and related subcommands.
2. `thorn serve` gateway plus the agency/runtime infrastructure it depends on.
3. Everything else.

The library-style API exists to serve the CLI and gateway. Do not preserve compatibility for `Agent`, `prompt()`, `@skill`, `@tool`, `wrap_function`, `ALL_BUILTIN_TOOLS`, or similar shapes unless the CLI or gateway depends on them.

## Required Commands

Use `uv` for every project command:

- Sync environment: `uv sync --all-extras`
- Full tests: `uv run pytest`
- Single test file: `uv run pytest tests/test_foo.py`
- Any Thorn command: `uv run thorn ...`

Do not use `pip install`, `python -m pytest`, or ad hoc project execution commands.

## Development Sequence

For non-trivial changes:

1. Read the relevant `AGENTS.md`, docs, code, and tests before designing.
2. Change architecture and module decomposition first when needed.
3. Change public module API surfaces and docstrings to reflect intended behavior.
4. Write or update black-box tests for the intended behavior.
5. Implement the behavior and supporting private code.
6. Run targeted tests first, then broader tests when shared behavior changed.
7. Self-review for Thorn-specific quality rules before final response or MR.

## Thorn Quality Rules

- Prefer explicit types over raw strings and integers for domain values.
- Avoid boolean parameters and fields unless the domain is truly binary and likely to remain so.
- Prefer class hierarchies over ad hoc tagged unions unless there is a clear local reason.
- Keep functions flat by handling early exits first.
- Use structured parsers or APIs instead of string manipulation where possible.
- Comments explain why, not what.
- Names favor clarity and accuracy over brevity.
- Keep acronyms consistently all caps or all lowercase in names: `UserID`, `user_id`, `HTMLNode`, `html_node`.

## Tests And Failures

- Add meaningful tests for changed behavior.
- Test the functionality that CLI/gateway users or module callers care about.
- If unrelated build/lint/test failures appear, investigate enough to classify them. Fix them when feasible; otherwise report concrete evidence and why they remain unresolved.
- Do not claim success without stating which `uv run ...` commands were run.

## Self-Review Checklist

Before handing back or opening an MR, use the account-level `self-review-before-mr` skill when available. At minimum:

- The change serves CLI/gateway priorities or explicitly justifies lower-priority work.
- No library-API compatibility shim was added without a load-bearing Thorn caller.
- Tests describe behavior rather than implementation details.
- Naming, types, and comments match the root `AGENTS.md` quality bar.
- The diff does not include unrelated churn.
- The diff is minimal and readable in the order a reviewer will encounter it.
- Any accumulated commits have been cleaned into focused review commits when practical.
- The branch has been fetched/rebased onto the remote base before MR creation when practical.
