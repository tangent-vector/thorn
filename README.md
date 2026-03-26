Thorn: A Tool for Building Agent Workflows
============================================

Thorn is a command-line tool and Python library that makes it easy to create agent workflows that flexibly mix deterministic code with AI prompts:

- Thorn allows you to easily write ordinary Python code that just happens to use AI for some of the steps.

- Thorn allows you to easily expose your Python code to agents as tools they can use. Tools you author with thorn are available not only to the Thorn agent, but also to other agents via Thorn's built in MCP server support.

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
def list_contributors_currently_active_on_slack() : list[str]
    """List the (human) project contributors who are currently online/active on Slack."""
    ...
```

Thorn uses Pydantic to expose the parameter signature and result type of your function as part of the generated tool description.

If your `@tool` function raises a Python exception, it will be surfaced to the agent as a tool-call failure.

#### Running Prompts From Python

Within any Python function, call `prompt` to prompt a fresh agent with a string.
Provide it an explicit list of the `tools` you want it to have access to:

```python
issues = prompt[list[str]]("""
    Build the project and provide a list of issues you can identify, if any.
    """,
    tools=[read_file,list_directory,build_project])
```

Thorn makes sure that the data returned by `prompt[T](...)` has the requested type `T`;
you don't need to do any extra work to get structured data like lists back from agents.

The agent that `prompt` runs will only have access to the tools you explicitly give it.
The tools can be any Python function decorated with `@tool` or `@skill`, including functions that Thorn provides for common tools like file reading.

Agents run by Thorn are able to report errors when they are unable to perform the requested task, and these are surfaced to Python code as `SkillError` exceptions.

#### Defining Skills

Mark a 

### The Tool

- Use `thorn run "..."` to execute a single prompt, or `thorn chat` to start an interactive CLI chat session.

- Thorn automatically locates any `.thorn/` directories in the current working directory or its ancestors, as well as any `.thorn/` directory in your user directory.

- Thorn loads 

Example
-------

We'll present a very simple example here, as a way of illustrating how Thorn can be used in a project.
Please note that this example is intended only as a sketch, and leaves out many details that would be needed for a real-world application.

For this example, imagine that we are developing a code project `my-app`.

### Adding Thorn Support to a Project

We start by adding a `.thorn/` directory to the root of the `my-app` repository (yes, yet another `.`-directory... if you can think of something better we're all ears):

```console
$ cd path/to/my-app/
$ mkdir .thorn
```

Then we can add ordinary Python code files under `.thorn/` that represent the tools, skills, etc. that we want to use for our project.

### A Simple Vibe Coding Loop

We can start by creating a simple vibe-coding loop that uses a policy of having a top-level `TODO.md` file listing tasks.
We'll use one prompt to pick the next task to execute, and another to actually execute the task, wrapping the whole thing in an infinite loop:

```python
from thorn import prompt, read_file, write_file, list_directory

def run_development_loop():
    while True:
        task = prompt[str]("""
            Read `TODO.md` and pick an incomplete task that seems to be unblocked.
            """,
            tools=[read_file, list_directory])
    
        prompt(f"""
            Complete the following development task and then mark it as done in `TODO.md`:

            {task}
            """,
            tools=[read_file, write_file, list_directory])
```

The `prompt` operation that Thorn provides 
If we want to improve this simple workflow, we can add a review loop:



Installing and Configuring
--------------------------

Thorn is a Python package, but is not currently distributed via any package registries.
The easiest way to start using Thorn is to clone this repository locally and install it via `pip`:

```console
$ cd path/to/thorn/
$ pip install -e .
```

Thorn uses a few environment variables to determine how it accesses your LLM provider:

- `OPENAI_API_URL`: the URL to your model provider (e.g., `https://api.openai.com/v1`)
- `OPENAI_API_KEY`: your access key with the provider
- `OPENAI_API_MODEL_NAME`: the name of the model you would like to use (e.g., `claude-4.6-opus-high`)

Thron currently only supports providers that expose an OpenAI-compatible web API.

If you have Thorn installed and configured, you should be able to run a simple test command like:

```console
$ thorn run "say hello, Thorn"
Hello! 👋 I'm Thorn, ...
```

