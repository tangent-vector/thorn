Thorn: A Tool for Building Agent Workflows
============================================

Thorn is a command-line tool and Python library for creating agent workflows that flexibly mix deterministic code with AI prompts:

- **Typed AI calls as a primitive**: Use `await prompt[list[str]](...)` to get structured data back from an LLM -- no parsing boilerplate, no schema wrangling, just the type you asked for.

- **Composable building blocks**: Tools, skills, and agent roles all nest naturally. A skill can call tools; a role aggregates skills; your Python code orchestrates everything.

- **Write tools once, use them everywhere**: Define project-specific tools in a `.thorn/` directory and they're instantly available to the built-in Thorn agent. Run `thorn serve` to expose the same tools to any MCP-compatible agent (Cursor, Claude Code, etc.).

Quick Start
-----------

Thorn is a Python package, but is not currently distributed via any package registries.
Clone this repository and install it locally:

```console
$ cd path/to/thorn/
$ pip install -e .
```

Thorn uses a few environment variables to determine how it accesses your LLM provider:

- `OPENAI_API_URL`: the URL to your model provider (e.g., `https://api.openai.com/v1`)
- `OPENAI_API_KEY`: your access key with the provider
- `OPENAI_API_MODEL_NAME`: the name of the model you would like to use (e.g., `claude-4.6-opus-high`)

Thorn currently only supports providers that expose an OpenAI-compatible web API.

Once installed and configured, verify with:

```console
$ thorn run "say hello, Thorn"
Hello! 👋 I'm Thorn, ...
```

Overview
--------

Thorn provides a toolbox rather than a singular solution, so we'll start by overviewing the most important utilities that Thorn provides.

### The Library

#### Defining Tools

Decorate an ordinary Python function with `@tool` to make it available to agents as a tool:

```python
@tool
def build_project() -> None:
    """Build the project. If the build fails, the response will include a summary of diagnostic messages."""
    ...

@tool
def list_contributors_currently_active_on_slack() -> list[str]:
    """List the (human) project contributors who are currently online/active on Slack."""
    ...
```

Thorn uses Pydantic to expose the parameter types and result type of your function as part of the generated tool description.

If your `@tool` function raises a Python exception, it will be surfaced to the agent as a tool-call failure.

#### Running Prompts From Python

Within any async Python function, use `await prompt(...)` to send a prompt to a fresh agent.
Provide it an explicit list of the `tools` you want it to have access to:

```python
issues = await prompt[list[str]]("""
    Build the project and provide a list of issues you can identify, if any.
    """,
    tools=[read_file, list_directory, build_project])
```

The `prompt[T](...)` syntax uses Python's subscript operator to specify the expected return type.
Thorn ensures that the value returned by `await prompt[T](...)` has the requested type `T`;
you don't need to do any extra work to get structured data like lists back from agents.

The agent that `prompt` runs will only have access to the tools you explicitly give it.
The tools can be any Python function decorated with `@tool` or `@skill`, including functions that Thorn provides for common operations like file reading.

Agents run by Thorn are able to report errors when they are unable to perform the requested task, and these are surfaced to Python code as `SkillError` exceptions.

#### Defining Skills

If you want to pull a `prompt`-based operation out as its own reusable function -- whether to call it from various places in your Python code, or to expose it as a tool to other agents -- you can use the `@skill` decorator:

```python
@skill(tools=[github_reading])
async def review_pull_request(pull_request_number: int) -> list[str]:
    """
Review pull request #{pull_request_number} and provide a list of
concerns or issues that should be addressed before it is committed.
If you approve of the pull request, then return an empty list.
"""
```

The `@skill` decorator gives the function an implementation that passes its docstring (with parameter values filled in) through to `prompt()`.
A `@skill` function can be called from your Python code like any other async function.

#### Defining Agent Roles

All of our prompting examples so far have had some clear limitations:

- No custom system prompts are being defined (only user prompts), and adding recurring/shared context to all the `prompt` calls in a project could result in tedious duplication.

- The available tools had to be stated explicitly for every `prompt` or `@skill`.

- All of the prompts have been one-and-done, with no persistent agent history. (Often this is actually a good choice, but being able to use history when it makes sense is also important.)

These limitations can be addressed by using Thorn to define custom agent *roles*.
An agent role is a Python `class` that extends `thorn.Agent`:

```python
from thorn import Agent

class MyProjectDeveloper(Agent):
    system_prompts = [
"""You are a developer working on `my-project`.

Your responsibilities are ...
"""
    ]
    tools = [
        read_file,
        write_file,
        list_directory,
        build_project,
        github_reading,
    ]
```

Roles can inherit from one another, and the complete system prompt and tool list for an agent is accumulated from all the roles it transitively inherits from.

A `@skill` decorator can be passed a `role=` argument to use that role's system prompts and tools:

```python
@skill(role=MyProjectDeveloper)
async def review_pull_request(pull_request_number: int) -> list[str]:
    ...
```

You can also construct agent instances directly and use their `prompt` method:

```python
developer = MyProjectDeveloper()
summary = await developer.prompt[str]("""
    Build the project and summarize any issues you find.
""")
```

Creating explicit `Agent` instances allows your code to make thoughtful decisions about when to retain history in a persistent agent, and when to start fresh.
For example, here is a loop that uses one long-lived agent to pick tasks and a fresh agent for each task:

```python
prioritizer = MyProjectDeveloper()
while True:
    task = await prioritizer.prompt[str]("""
Pick a development task from `TODO.md` that makes sense to do next.
""")
    developer = MyProjectDeveloper()
    await developer.prompt[str](f"""
Do the following development task:
{task}
""")
    ...
```

### The Tool

The `thorn` command-line tool can help you run the tasks and workflows you've defined.

#### Basic Usage

Use `thorn run "..."` to execute a single prompt, or `thorn chat` to start an interactive CLI chat session.

#### Providing Tools and Skills

When you run `thorn`, it automatically searches for any `.thorn/` directories in the current working directory (or its ancestors), as well as any `.thorn/` directory in your user home directory.
Any `.py` files in a `.thorn/` directory are automatically loaded, and any `@tool` or `@skill` functions defined in them will be available to the top-level Thorn agent.

As an example, if you have defined a file `.thorn/dev_tools.py` in your repository, and it contains:

```python
@tool
def build_project() -> None:
    """Build the project. ..."""
    ...

@tool
def run_tests() -> None:
    """Run the test suite. ..."""
    ...
```

Then you should be able to prompt `thorn` and have it use those tools:

```console
$ thorn run "build and test the project, and summarize any failures"
...
```

#### Serving Tools to Other Agents via MCP

While you can use `thorn` directly as a coding agent, it is a small tool and not as full-featured as popular coding agents like Cursor or Claude Code.
Luckily, you don't have to choose; you can define project-specific tools using Thorn and serve them to your preferred agent via MCP:

```console
$ thorn serve --transport streamable-http
```

This means you can define a `build_project` tool in your `.thorn/` directory once, and immediately use it from Cursor, Claude Code, or any other MCP client -- without any of those tools needing to know how your project's build system works.

The `--transport` option selects between `streamable-http` or `stdio`, depending on what your MCP client supports.

Example Project
---------------

The `examples/calc/` directory demonstrates a non-trivial Thorn setup: a C++ calculator project with `.thorn/` tools for building, testing, and a multi-agent development workflow with distinct `Architect`, `APIDesigner`, `Implementer`, and `TestEngineer` roles.
It's a good starting point for understanding how the pieces fit together in practice.
