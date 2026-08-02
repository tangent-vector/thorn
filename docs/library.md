Thorn Python Library
====================

Thorn can also be used as a Python library for building agent workflows
that flexibly mix deterministic code with AI prompts.

> **Note:** The library API described here is functional but is not the
> current development focus. The primary use case for Thorn today is the
> [gateway](../README.md), which deploys autonomous agents that monitor
> and respond to activity on a code forge (GitHub, GitLab).

Using Tools
-----------

The current supported tool path is Thorn's built-in tool catalog plus
sandboxed scripts and MCP servers.  `thorn run`, `thorn chat`, and the
gateway do not auto-load arbitrary Python files from a project checkout.

For Python `prompt(...)` calls, pass built-in tool functions from
`thorn.tools` or other first-party toolsets.  Thorn accepts those known
built-ins because the brain process and toolhost daemon both know how
to route them across the sandbox boundary.

```python
from thorn import prompt
from thorn.tools import list_directory, read_file

files = await prompt[list[str]](
    "List the important source files in this project.",
    tools=[read_file, list_directory],
)
```

Arbitrary user-defined Python callables are not accepted as prompt
tools.  The older `@tool` decorator remains the low-level marker used
by Thorn's built-in tool catalog and by the legacy
`thorn.core.discover_tools()` helper, but it is no longer the normal
extension mechanism for CLI or gateway agents.

For project-specific deterministic behavior, put ordinary scripts in
the repository and let the agent run them with its sandboxed
`run_shell` tool.  For reusable agent guidance, create
`.agents/skills/<name>/SKILL.md` files that tell the agent which
commands, scripts, policies, or repository conventions to use.  For
structured external tools, configure `.agents/mcp.json` files that the
context pipeline can discover instead of importing Python callables into
the brain process.

Running Prompts From Python
---------------------------

Within any async Python function, use `await prompt(...)` to send a prompt to a fresh agent.
Provide it an explicit list of the `tools` you want it to have access to:

```python
from thorn.tools import list_directory, read_file

issues = await prompt[list[str]]("""
    Inspect the project and provide a list of issues you can identify, if any.
    """,
    tools=[read_file, list_directory])
```

The `prompt[T](...)` syntax uses Python's subscript operator to specify the expected return type.
Thorn ensures that the value returned by `await prompt[T](...)` has the requested type `T`;
you don't need to do any extra work to get structured data like lists back from agents.

The agent that `prompt` runs will only have access to the tools you
explicitly give it.  Those tools must be Thorn built-ins; ordinary
user-defined Python functions are not accepted by the sandbox-era tool
policy.

Thorn's API is async. For synchronous scripts, wrap your workflow with `thorn.run()`:

```python
import thorn
from thorn import prompt
from thorn.tools import list_directory

async def main():
    issues = await prompt[list[str]](
        "List likely code issue areas.",
        tools=[list_directory],
    )
    print(issues)

thorn.run(main())
```

Defining Skills
---------------

If you want to pull a `prompt`-based operation out as its own reusable
function for Python workflows, you can use the `@skill` decorator:

```python
from thorn import skill
from thorn.tools import read_file

@skill(tools=[read_file])
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
from thorn.tools import list_directory, read_file

class MyProjectDeveloper(Agent):
    system_prompts = [
"""You are a developer working on `my-project`.

Your responsibilities are ...
"""
    ]
    tools = [
        read_file,
        list_directory,
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
Pick a development task from `ROADMAP.md` that makes sense to do next.
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

Use `thorn run "..."` to execute a single prompt, or `thorn chat` to
start an interactive CLI chat session. Both commands create a fresh
local CLI session by default and persist it under the local agency
(`~/.thorn`, or the directory passed with `--agency`).

Use `thorn sessions list` to inspect persisted local CLI sessions.
Use `thorn chat --resume <session-key>` to re-enter one interactively,
or `thorn run --resume <session-key> "..."` to append a non-interactive
follow-up turn. A resumed session keeps the workspace path stored in
the session metadata; an explicit `--workspace` must match that stored
path, and resume fails if the stored workspace no longer exists or the
session is already locked by another Thorn process.

### Providing Context, Skills, and Scripts

`thorn run` and `thorn chat` use the local CLI agent's built-in file
tools and sandboxed shell tool.  They also gather repository and
agency context such as `AGENTS.md`, `MEMORY.md`, journals,
`.agents/skills/<name>/SKILL.md`, and MCP configuration files.

They do not automatically import `.agents/thorn/*.py`.  If a project
needs repeatable behavior, add a script or ordinary project command
and describe it in a skill or `AGENTS.md`.  For example, a project can
ship `scripts/check` and a `.agents/skills/project-check/SKILL.md`
that tells the agent when to run it.

Then you can prompt `thorn` and have the agent use those commands
through the existing sandboxed shell tool:

```console
$ thorn run "build and test the project, and summarize any failures"
...
```

### Serving Tools to Other Agents via MCP

You can serve Thorn's built-in tool catalog to another MCP client:

```console
$ thorn serve mcp --transport streamable-http
```

This command exposes the same first-party Thorn tools that the toolhost
registry knows how to route.  It does not import project
`.agents/thorn/*.py` files or serve repository skills as Python tools.
Use a normal MCP server when a project needs a custom structured tool
surface for external clients.

The `--transport` option selects between `streamable-http` or `stdio`, depending on what your MCP client supports.
