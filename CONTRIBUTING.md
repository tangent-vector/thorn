# Contributing to Thorn

Thorn is an experimental, independently maintained project. Contributions are
welcome when they are aligned with the project's current CLI- and
gateway-focused direction and are reasonably scoped for review.

## Discuss the change first

Before investing in an implementation, open an issue that explains the problem,
the user-visible outcome, and the proposed scope. Please wait for agreement on
the direction before opening a pull request. This helps avoid asking
contributors to spend time on changes that do not fit Thorn's priorities or
current architecture.

Keep each issue and pull request focused on one coherent outcome. Separate
unrelated cleanup, refactoring, and feature work even when they touch nearby
files.

The end-user priorities are, in order:

1. The `thorn` CLI, including `thorn run`, `thorn chat`, and agency
   administration.
2. The `thorn serve` gateway and the agency/runtime infrastructure it needs.
3. Supporting components and experiments.

The Python library surface exists to support the CLI and gateway. It is not a
stable compatibility contract, so preserving or expanding it for external
embedding is not by itself a project goal.

## Development workflow

Thorn uses `uv` for its environment and project commands:

```console
$ uv sync --all-extras --locked
$ uv run pytest
$ uv run ruff check .
$ uv build
```

Use targeted tests while iterating, then run the full relevant validation before
requesting review. Add tests for behavior users or operators care about; do not
add tests whose primary purpose is to freeze documentation wording.

## Pull requests

A pull request should:

- link the agreed issue, normally with `Closes #...`;
- explain the user-visible outcome and important design choices;
- include meaningful tests for changed behavior;
- update current documentation when commands, configuration, or guarantees
  change; and
- state the exact validation commands that passed.

Do not include credentials, private deployment details, generated traces, or
unrelated formatting churn.

Potential security vulnerabilities should not be discussed in a public issue.
Follow the private reporting process in [SECURITY.md](SECURITY.md) instead.
