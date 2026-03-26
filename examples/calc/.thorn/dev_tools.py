"""Build tools for the calc example project.

These are auto-discovered by thorn from the .thorn/ directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import thorn
from thorn import tool, prompt, skill

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "src"

@tool
async def add_module(name: str, description: str) -> None:
    """Add a new module to the project, given a name and brief description."""

    await prompt(f"""
Add a new module named `{name}` to the project, comprising a paired header and source file under the `src/` directory.
If somehow corresponding files already exist, do not overwrite them, and instead raise an error to ensure the situation is brought to the user's attention.

You should fill in the header file with an initial comment based on user-provided description:

{description}

Do NOT include additional content in the header file. Your only task is to put the files into place.

The source file should include the header file, and should be empty otherwise.
""",
        tools=[thorn.read_file, thorn.write_file, thorn.list_directory])

@skill(tools=[thorn.read_file, thorn.list_directory])
async def list_submodules(name: str) -> list[str]:
    """
List the names of the sub-modules of the module `{name}`.

You should use the content of the header and/or source file for the module
to determine what its sub-modules are, from the architecture description given.
"""

@tool
async def define_architecture(name: str) -> str:
    """Define the architecture for a module."""

    await prompt(f"""
You are responsible for defining the architecture for the module `{name}`.
The source and header files for the module should already be in place under `src/`;
if they are not, you should raise an error to ensure the situation is brought to the user's attention.

In the case where the module is `main`, you should take responsibility for the `main.cpp` file,
and there will (almost always) be no corresponding header.

Be aware that your responsibilities are very narrow. You should:

  - Read and respect any existing comments or other content in the files you are
    responsible for. If there are existing design decisions laid out in the comments
    for your module, then assume those represent the user's intentions and do not
    override them.

  - Ensure the header file for the module (or `main.cpp`) starts with a comment block
    that clearly states the purpose, responsibilities, and requirements of the module.

  - Describe how the module is decomposed into sub-modules, if any, and for
    each sub-module, ensure that its files have been created using the `add_module` tool.

    When in doubt, assume that even simple programs should be decomposed into modules,
    so long as there are clearly distinct responsibilities or concerns that can be defined.
    
    If you are being invoked on a module that has well-written comments explaining its purpose,
    but seemingly has no sub-modules, then you should assume you are responsible for creating
    appropriate sub-modules, unless you are exceedingly confident that the implementation
    of the module would comprise less than 50-100 lines of code.

  - Describe the dependencies of the module, whether on other modules in the project,
    standard libaries, or third-party libraries, and ensure that correct include directives
    are added to the header and/or source file. Dependencies that are expected to impact
    the public API of the module should be documented and included in the header file.
    Dependencies that are expected to only be relevant to the implementation of the module
    should be included in the source file only, and may be documented in the source file
    rather than the header file, if they are best thought of as implementation details only.

Do NOT write any code, whether that's declarations, definitions, etc.

Do NOT be overly prescriptive about the architecture and design of your sub-modules.
If you are listing specific names for types, functions, etc. that you expect a sub-module to define or provide,
then you are trying to do more than your job description allows.
""",
        tools=[thorn.read_file, thorn.write_file, thorn.list_directory, add_module])

@tool
async def fully_define_architecture(name: str) -> str:
    """Define the architecture for a module and its sub-modules, recursively."""

    await define_architecture(name)
    submodules = await list_submodules(name)
    for submodule in submodules:
        await fully_define_architecture(submodule)
