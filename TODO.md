- Validation feedback is appended to tool results via `ValidationTracker`, but validation is only triggered when workflow tools explicitly record results. Consider whether validation should be triggered automatically in response to file writes (or other actions), rather than requiring explicit opt-in from each tool.

- survey current built-in tools and make sure they are following industry best practices

- `thorn do` instead of `thorn run`, and then rename `thorn.prompt` over to be `thorn.do` (and similarly, `Agent.prompt` becomes `agent.do`)

- In the workflow: a "clarify and then delegate" role/step/tool that the concierge can invoke, that works to explore the project and (optionally) grill the user to get clarification on their intent, before moving on to actually delegating to a `coordinator` agent to get the work done.

- Consider splitting `@skill` so that there's a distinction between "a function whose implementation is a prompt" and the exposure of such a function to the rest of the system

- The `ask_user` tool should fail gracefully (or be unavailable) when stdin is not a tty, so that non-interactive `thorn run` sessions don't hang waiting for input that will never come.

- Some sort of POR around how to fit approval into all this, by having a notion of tools that should require approval (or maybe have filters/predicates to decide when they need approval)

- Consume typical definitions of skills, slash commands and personas (e.g., like in `.claude/`)

- Consider scraping MCP servers, skills, etc. from `.cursor/`, `.claude`, etc. rather than just from `.thorn/`... or at least allowing configuration in `.thorn/` to specify that additional configuration stuff should be loaded from such a directory

- Make it easier for code under `.thorn/` to reference text/markdown documents in the same directory as part of their prompts, etc. (or just document the conventions for getting at resources).

- Make it easier to expose tools that don't require writing Python (e.g., stuff that should just amount to running a command line or shell script)

- Expose additional predefined tool sets beyond the existing `FILE_READING` and `FILE_WRITING`: a `file_manipulation` set (reading plus writing/creating directories), and a `web_research` set (web searches via something like duckduckgo, plus beautifulsoup or similar for extracting content).

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

- Per-agent cost/token tracking breakdown (aggregate tracking already works via `UsageTracker`), and monetary cost estimation

- Collapsible terminal output using ANSI folding sequences (the same mechanism GitHub Actions uses for grouped log lines), so tool call details can be expanded on demand

- A `thorn trace view` command that opens a local web page to visualize JSONL trace files (produced by `--trace`) as an interactive execution tree with timing

- The LLM provider (`_provider.py`) does not set `max_tokens` on API requests, so a degenerate model response can produce unbounded output. The agent loop (`_loop.py`) also has no repetition detection. Together these allow a stuck LLM to spin forever repeating tokens. Short-term: add a configurable `max_tokens` to `OpenAIProvider.complete`. Longer-term: add repetition/loop detection in the agent loop itself.

- Terminal output from `thorn run` can be truncated when the process completes, making it hard to diagnose issues from the log alone. Consider flushing/syncing output before exit, or writing a separate structured trace log.

- Allow a Python module to be used in a `tools=[...]` list, akin to how we allow iterables of tools. A Python module in such a list would stand in for the list of `@tool` functions in the `__all__` of that module.

- The `ls`-equivalent tool provided by Thorn (which should probably be a glob-like tool, in practice) should support giving compact "preview" information on files/directories, using file-format-specific policies (perhaps eventually pluggable, but let's not worry about that yet).
  One concrete example would be surfacing the title of a Markdown document, when available (a leading level-1 heading). We can imagine that listing the contents of `docs/` might yield:

  ```
  docs/
    README.md       "The Foo Project"
    conventions.md  "Coding Conventions"
    guide/          "Foo User's Guide"
  ```

  In the above example, the tool automatically scraped the leading heading from `README.md` and `conventions.md` to intuit their titles, and then also identified that `docs/guide/` contained its own `README.md`, and applied a policy  to infer an appropriate hint/title for the `guide/` directory from that.

  As a possibly-questionable extension/abuse of that concept, we could make it so that when the tool sees a `SKILL.md` file it extracts the `description:` from the YAML front-matter and uses that as the hint text intead of trying to scrape for a title in the Markdown content. With that kind of subtle policy tweak, a simple `ls`-like tool call on `.agents/skills/` directory would "automatically" yield a listing of available skills and their descriptions.

- Some kind of config file under `.thorn/` that can be used to specify options even for sub-tools (like the gateway server).
