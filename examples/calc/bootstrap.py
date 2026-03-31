"""Bootstrap script for the calc example project.

Resets the project to a clean starting state by:
1. Removing existing src/, build/, and tests/ directories
2. Copying template files into place (including doctest.h)
3. Verifying .thorn/ tools exist

Run from the repo root or from examples/calc/:
    python examples/calc/bootstrap.py
    python bootstrap.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PROJECT_DIR / "template"
SRC_DIR = PROJECT_DIR / "src"
BUILD_DIR = PROJECT_DIR / "build"
TESTS_DIR = PROJECT_DIR / "tests"
THORN_DIR = PROJECT_DIR / ".thorn"


def main() -> None:
    print(f"Bootstrapping calc project in {PROJECT_DIR}")

    for d in (SRC_DIR, BUILD_DIR, TESTS_DIR):
        if d.exists():
            shutil.rmtree(d)
            print(f"  removed {d.relative_to(PROJECT_DIR)}/")

    for src_path in TEMPLATE_DIR.rglob("*"):
        if src_path.is_file():
            rel = src_path.relative_to(TEMPLATE_DIR)
            dst = PROJECT_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            print(f"  copied template/{rel}")

    expected_tools = ["build_tools.py", "module_tools.py", "orchestration.py", "roles.py"]
    for name in expected_tools:
        tool_path = THORN_DIR / name
        if tool_path.exists():
            print(f"  verified .thorn/{name}")
        else:
            print(f"  WARNING: .thorn/{name} not found!")

    print("\nProject is ready. Next steps:")
    print('  cd examples/calc')
    print('  thorn run "coordinate \'implement a calculator\'"')


if __name__ == "__main__":
    main()
