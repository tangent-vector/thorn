Thorn Python Library
====================

Thorn can also be used as a Python library for building agent workflows
that flexibly mix deterministic code with AI prompts.

> **Note:** The library API described here is functional but is not the
> current development focus. The primary use case for Thorn today is the
> [gateway](../README.md), which deploys autonomous agents that monitor
> and respond to activity on a code forge (GitHub, GitLab).

Defining Tools
--------------

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

Running Prompts From Python
---------------------------

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

Thorn's API is async. For synchronous scripts, wrap your workflow with `thorn.run()`:

```python
import thorn
from thorn import prompt

async def main():
    issues = await prompt[list[str]]("List code issues.", tools=[build_project])
    print(issues)

thorn.run(main())
```

Defining Skills
---------------

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
When an agent is unable to perform the requested task, it raises a `SkillError`, which you can handle like any other exception:

```python
for pr in open_pull_requests:
    try:
        concerns = await review_pull_request(pr.number)
        if concerns:
            await post_review_comments(pr.number, concerns)
    except SkillError as e:
        await notify_team(f"Could not review PR #{pr.number}: {e.detail}")
```

Defining Agent Roles
--------------------

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

A `@skill` decorator or a `prompt` call can be passed a `role=` argument to use that role's system prompts and tools:

```python
@skill(role=MyProjectDeveloper)
async def review_pull_request(pull_request_number: int) -> list[str]:
    ...
```

```python
issues = await prompt[list[str]]("""
    Build the project and provide a list of issues you can identify, if any.
    """,
    role=MyProjectDeveloper)
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

The CLI
-------

### Basic Usage

Use `thorn run "..."` to execute a single prompt, or `thorn chat` to start an interactive CLI chat session.

### Providing Tools and Skills

When you run `thorn`, it automatically searches for `.agents/thorn/` directories in the current working directory (or its ancestors).
Any `.py` files in a `.agents/thorn/` directory are automatically loaded, and any `@tool` or `@skill` functions defined in them will be available to the Thorn agent.

As an example, if you have defined a file `.agents/thorn/dev_tools.py` in your repository, and it contains:

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

### Serving Tools to Other Agents via MCP

You can define project-specific tools using Thorn and serve them to your preferred agent via MCP:

```console
$ thorn serve mcp --transport streamable-http
```

This means you can define a `build_project` tool in your `.agents/thorn/` directory once, and immediately use it from Cursor, Claude Code, or any other MCP client -- without any of those tools needing to know how your project's build system works.

The `--transport` option selects between `streamable-http` or `stdio`, depending on what your MCP client supports.
