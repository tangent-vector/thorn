- `thorn do` instead of `thorn run`, and then rename `thorn.prompt` over to be `thorn.do` (and similarly, `Agent.prompt` becomes `agent.do`)

- In the workflow: a "clarify and then delegate" role/step/tool that the concierge can invoke, that works to explore the project and (optionally) grill the user to get clarification on their intent, before moving on to actually delegating to a `coordinator` agent to get the work done.

- Consider splitting `@skill` so that there's a distinction between "a function whose implementation is a prompt" and the exposure of such a function to the rest of the system

- Tools or other support to allow querying the human user as part of a workflow (and making sure those queries are surfaced in a way that makes them fit in naturally during a `thorn chat` or `thorn run` session... and that they fail if stdin doesn't appear to be a tty).

- Some sort of POR around how to fit approval into all this, by having a notion of tools that should require approval (or maybe have filters/predicates to decide when they need approval)

- Ensure convenience in accessing MCP-based skills from within Python

- Consume typical definitions of skills, slash commands and personas (e.g., like in `.claude/`)

- Consider scraping MCP servers, skills, etc. from `.cursor/`, `.claude`, etc. rather than just from `.thorn/`... or at least allowing configuration in `.thorn/` to specify that additional configuration stuff should be loaded from such a directory

- Make it easier for code under `.thorn/` to reference text/markdown documents in the same directory as part of their prompts, etc. (or just document the conventions for getting at resources).

- Make it easier to expose tools that don't require writing Python (e.g., stuff that should just amount to running a command line or shell script)

- Allow `tools=` to support iterables of tools in the list alongside individual tools, so that users can easily write a shorthand for a list/set of tools. Thorn should probably expose basic `file_reading` (read files, list directories, grep, globbing search), `file_manipulation` (`file_reading` plus the ability to write files and create directories), `web_research` (web searches via something like duckduckgo, plus beautifulsoup or similar for extracting the content).

- Provide a centralized mechanism for file permissions management, so that agents roles can include explicit opt-in or opt-out access to specific files, directories, etc. (probably using notation similar to `.gitignore`). All file access through Thorn's built-in file toolset should check for permissions according to those rules.

  Note that having read-write *permissions* to a file path doesn't mean writing is automatically possible, since an agent also needs access to the `write_file` tool.

  The default `Agent` class should probably provide a default of write access to `.` (meaning the "workspace" directory, whatever that should be). (The concept of the "workspace" directory for Thorn should probably default to the CWD at the time `thorn` was launched, but its also possible it should support defaulting to the deepest enclosing directory with a `.thorn/` directory under it)

- Thorn should probably read and respect any `AGENTS.md` file(s) that are set up in a project, using the conventions established by other tools. The content of those files should be piped into `thorn`s agents as additional system-prompt content.

- We have a potential gotcha lurking, in that `@skill` functions use the docstring to represent the prompt, which leaves them without a way to convey all the things that a `@tool` function uses its docstring to set up. It may be that `@skill` is too 'clever" for its own good, and we should instead just tell people to write an ordinary function that calls `prompt()` instead.

- Having custom `Agent` subclasses immediately opens the door to wanting to define the equivalent of `@skill` methods on them (assuming we don't just eliminate `@skill` as described above).

- We're still missing functionality to do something like a Python `with` to establish scoped context that will be visible to any Thorn agents within that context/scope (perhaps with a filter on their role/class, so that we don't overwhelm agents with context that shouldn't be relevant to them).

- Concurrency/parallelism is a clear opportunity, but also a risk. It makes sense to want to fire off multiple prompts and let them proceed independently (and `asyncio` makes that relatively simple in principle, even within a single process)

- Observability of the state of a given running `thorn` instance is something that would be nice, at least for debugging purposes. How can I get a good idea of what agents are up and running?

- There's a never-ending list of things we could try to adapt from more advanced agent tools, such as being able to have multiple ongoing/persistent chat sessions that you can attach/detach, etc.

- Support for git worktrees seems like one of the most important things to get right, sooner or later. It's not really something you want users to have to implement for themselves in user space, over and over.

- Support using "slash command" syntax when using `thorn run` or `thorn chat` so that a user can be completely explicit about their intention to run a specific tool/skill

- Rich `Live` display for real-time spinners/progress indicators during tool execution, replacing the completed-tool line in-place

- Rich `Tree` rendering for a post-run execution summary (the full agent/tool call tree printed after the run completes)

- Cost/token tracking per agent and in aggregate, sourced from the provider's response metadata (token usage fields in the OpenAI streaming API)

- Collapsible terminal output using ANSI folding sequences (the same mechanism GitHub Actions uses for grouped log lines), so tool call details can be expanded on demand

- A `thorn trace view` command that opens a local web page to visualize JSONL trace files (produced by `--trace`) as an interactive execution tree with timing

- The LLM provider (`_provider.py`) does not set `max_tokens` on API requests, so a degenerate model response can produce unbounded output. The agent loop (`_loop.py`) also has no repetition detection. Together these allow a stuck LLM to spin forever repeating tokens. Short-term: add a configurable `max_tokens` to `OpenAIProvider.complete`. Longer-term: add repetition/loop detection in the agent loop itself.

- Terminal output from `thorn run` can be truncated when the process completes, making it hard to diagnose issues from the log alone. Consider flushing/syncing output before exit, or writing a separate structured trace log.

- Validation is currently skipped for non-implementer roles (architect, api_designer, test_engineer) as a pragmatic workaround. Longer-term options: (a) an `implement_stubs` tool for the api_designer so the build can pass after API design, (b) context-dependent validation decisions made by coordinators, (c) a general mechanism for coordinators to specify which validation rules apply per-delegation.

