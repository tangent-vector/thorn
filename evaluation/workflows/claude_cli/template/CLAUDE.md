# Project Guidance

## Design principles

- Employ clean modular decomposition with well-defined APIs between components.
  Each module should have a clear responsibility and a minimal public interface.
- Prefer flat code over deeply nested code.
  Handle early-out cases first so the main logic path stays at a low nesting level.
- Define explicit types rather than relying on primitive types or strings to
  carry semantic meaning.

## Testing

- Ensure that all non-trivial functionality is covered by tests.
  Tests should exercise the public API of each module, not internal details.
- Do not consider your work complete until all tests pass.

## Building and running tests

This project uses CMake. Test files live under `tests/` and are named `*_test.cpp`.
The test framework is [doctest](https://github.com/doctest/doctest) (header-only,
already provided in `tests/doctest.h`).

To configure, build, and run tests:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run this sequence before reporting that your work is complete, and fix any
build errors or test failures you find.
