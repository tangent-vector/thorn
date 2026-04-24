Please look at @\home\tfoley\.cursor\plans\sandbox_phase_c_plan_7f0b9d0f.plan.md. You and I are going to put together a clear design for the prerequisite work that plan refers to a "phase C.0".

A few pieces of context-setting preamble:
- It is okay if work we do breaks the old library API contract for Thorn. Library use of Thorn is no longer a feature we are trying to support, with the only caveat being that whenever the actual runtime/agency/gateway implementations use features of the old Thorn library API (`@tool` decorators, `prompt()` calls, `Agent` subclasses, etc.) we need to make sure that we get that code working as expected/required on top of whatever the new infrastructure looks like. The key point is just that we have no concern for library use cases any more and, by extension, `thorn serve mcp` is okay to break (since it exists to serve tools authored using Thorn's library features).
- Any work we do from here on should aim for the aspirational goal of reducing the differences between the CLI (`thorn chat`, `thorn run`) and gateway (`thorn serve`) use cases and their execution flows. At present, the core idea is that each of these cases loads a `Runtime`/agency from its on-disk configuration, and then hosts some number of agents and sessions. Things like tool calling, system-prompt construction, etc. should ideally work as similarly between the CLI and gateway cases as we can make work.
- Backwards compatibility with existing data (e.g., existing agency configurations/state that have been saved off somewhere) is not a requirement. Inserting code to be compatible with previous schemata or file/directory layouts, or to migrate data over to new expectations is not worth the time if it adds anything to the overall complexity budget.

The high-level goal for what you and I are going to design is that when the agency/runtime infrastructure is about to prompt a session (e.g., because of unhandled notifications in its inbox) we run a single unified code path that collects relevant context for the session, based on its home/workspace paths, the home/workspace paths of its parent agent and, potentially, paths related to the agency/runtime itself.

The notion of "relevant context" here is a catch-all for things that will contribute to the system prompt for the session, including:
- Agent policy/behavior guidance, as typically given in `AGENTS.md` files
- Agent memory overview information, as typically given in `MEMORY.md` files
- Temporal memory information, sourced from the session's journal files
- Lists of available tools, which could include MCP-server-based tools discovered via `.agents/mcp.json` files, perhaps plain Python callables in a `.agents/thorn.py` or similar file, etc.
- Lists of available skills, sourced from `.agents/skills/` directories

(Note that in every place where we refer to `AGENTS.md` we are effectively referring to either a file actually named `AGENTS.md` or alternative names like `CLAUDE.md`, etc. Similarly, when referring to things under a `.agents/` directory, we might also search a `.claude/`, `.cursor/`, etc. directory for comparable information)

Additional kinds of context that might be gathered in the same overall pass can include:
- Relevant `.aiignore` files, which can establish constraints on what file paths the built-in file-reading/manipulation tools will ignore in tool calls that result from the current round of prompting
- Other information that we might decide to add over time, such as identifying whether the session's workspace directory is under a git repository (by noting any `.git` directories) so that we include a relevant "dashboard"/status updater as part of the current round of prompting.

From an end-user-visible standpoint, there are two main factors that come together to determine the context that gets gathered for a given session:
- Which directories the system considers relevant to that specific session
- Which files or other information the system looks for, relative to each of those directories

Note that the framing there intentionally emphasizes that the context-gathering process should treat any/all directories it gathers from as equivalent, and look for all the same files/directories relative to each of them. In any cases where the desired policy deviates from that goal of consistency, we should call it out (and also question whether it's really a necessary break).

In order to understand what directories should be considered when gathering context for a session, it is important to understand the main kinds of directory paths that we traffic in inside a Thorn runtime (the codebase may not be consistent about how it names/uses these, but my descriptions here should be considered as establishing the intention):

- The *agency* has an *agency home* directory (the one that contains `gateway.json`... which should really be named `agency.json`).

  For the gateway use case, an agency also has an *agency workspace* directory, configured via a setting in `gateway.json`.

  It is less clear what the agency workspace directory should/could be in the CLI (aka "local gateway") case, so that's a confounding factor (not technically on our plate to address right now).

- A given *agent* has its *agent home* directory: `<agency-home>/agents/<agent-id>/home/`.

  For the gateway use case, an agent also has its *agent workspace* directory: `<agency-workspace>/agents/<agent-id>/workspace/`

  For the CLI use case, it isn't entirely clear that an entire *agent* should have a workspace directory. As with the agency, this is a bit of a confounding factor.

- A given *session* inherits the agent home directory as its *session home* directory.
  A session additionally has a *session-key home* directory (I'm making up a term here...) which is `<agent-home>/<session-key>`.

  (Note that when `<session-key>` appears in a path, the correct interpretation is that the `/` separators in the session key map to directory separators; they should not get escaped (as I believe the implementation currently does in some cases...))

  Every session has a *session workspace* directory.
  For the gateway use case, the session workspace is `<agent-workspace>/<session-key>`.
  For the CLI use case, the session workspace for a session is always the CWD when the `thorn run` or `thorn chat` command was invoked (even if connecting to an existing session).

  (A simplistic way to look at things is that when using the CLI, *all* workspace paths (agency, agent, and session) are kind of overriden to just be the CWD of the `thorn chat` invocation, for the purposes of that session)

  Aside: the current logic for file-access rules is probably going to need to be thrown away, but it is important to note that for every session, it should have R/W access to the entire agent home directory and to the entire agent workspace (even when the *session* workspace is a subdirectory of the agent workspace). All `.aiignore` files should be interpreted/understood relative to the directory where the specific `.aiignore` file appears, for the purposes of ignoring/hiding files.

With the definitions of the relevant directories out of the way, we can now define which directories should be *considered* as sources of context for a given session about to be prompted:

- The session workspace directory is considered.

- If the agent workspace directory is present and not the same as the session workspace directory, then each directory along the parent (`..`) chain from the session workspace to the agent workspace directory is considered.

  (It is an exceptional situation if we have an agent workspace that doesn't contain the session workspace, but I suppose we need a plan in case that happens)

- Similar to the previous two notes, every directory on the `..` chain from the session-key home to the agent home path is considered.

  This collection of things from the session-key home is intended as a way to ensure that topical memory/skills/etc. can be organized under the agent's home directory and automatically brought into scope for sessions where they are appropriate.

  This kind of collection could conceivably be extended to support gathering context from additional paths under the agent home, based on other session-key templates (see `coordination.md` for background) that are determined to be relevant to the current session.

- Finally, as a possible extension, the `<agency-home>/agents/<agent-id>/` directory can be considered. Note that this is *not* the agent home directory, and once we are doing container isolation this location is importantly outside of what the agent itself can see or modify (only the `home/` sub-directory gets mounted into the container).
  The idea of including this location is to allow the human operators of the agency to inject context that *must* be included, no matter what the agent does.

For each directory to be considered, we want to apply more or less the same rules for gathering relevant context ("relevant context" as defined earlier).
There are some corner cases where we might only collect certain kinds of context from certain of the cases above, but these details could be left for the future:

- It might make sense to only load `MEMORY.md` files from locations under the agent's home directory (notably: not from the workspace)

- If we decide to have additional categories of "all caps context files" distinct from `MEMORY.md` and `AGENTS.md` (where `CLAUDE.md`, etc. are just fallback alternatives to be considered when `AGENTS.md` is not found in a given directory), then we would want to consider where it makes sense to allow them.
  E.g., an OpenClaw-style `SOUL.md` or some other kind of `PERSONA.md` might not want to allow contributions from the workspace.

- It might make sense to only consider `.aiignore` files coming from locations under the agent's workspace directory (this may not matter, if scraping for `.aiignore` files is something we incorporate into the file reading/manipulation tools directly, rather than treating as a prompt-time context gathering step... which... now that I write it down... yeah... `.aiignore` handling is for the file-related tools to deal with, and not a context-gathering concern)

The ordering of contributions is important, when it comes time to assemble the full system prompt.
At the most basic, we can think in terms of a hierarchy from outer-most to inner-most context, along the lines of:

- The `<agency-home>/agents/<agent-id>/`, if included in consideration, is the outer-most scope currently being proposed/discussed

- The directories under the agent home are next, with the agent home being the outer-most of these, and the session-key home being the inner-most.

- The directories under the agent workspace are next, with the agent workspace being the outer-most, and the session workspace being the inner-most.

I think an algorithm for gathering relevant context should probably proceed in two well-defined steps:

- First, identify the full sequence of (unique) directories to consider, in order from outer-most to inner most.
  If desired, these directories can be tagged with a kind if we decide we need to have different rules for what to collect from, e.g., home vs. workspace directories.

- Second, iterate over those "directories to consider," and for each check for the presence of each kind of relevant context and assemble a per-directory structure that collects all the context contributions from that directory "layer"

  (To be clear, I'm explicitly advocating for what amounts to an array-of-structs style of data structure instead of structure-of-arrays, with the intention that doing so makes it easier for us to modify/extend the categories of relevant context over time, as we learn what makes sense to include)

The system prompt is then constructed in terms of a few different blocks, each of which may source content from the collected per-directory information described above:

- Any "super-global" prompting that is coming from the Thorn system itself should go first (obviously).
  This would probably include any necessary guidance on how to interact with the messaging system, how multi-session agents work in Thorn, how the filesystem is being used (agent home vs. workspace, etc.), and how to use and interact with memory.

- If we allow for custom per-agent `SOUL.md` or similar, then all relevant contributions of that kind go here, from outer-most to inner-most.

- All contributions to agent policy (e.g., `AGENT.md` files) go next, from outer-most to inner-most.

- A list of all skills (accumulated outer-to-inner) comes next, with each entry providing a path where the session can access the content of the skill, plus the description sourced from the `SKILL.md` frontmatter.

- All contributions to agent memory (`MEMORY.md`) go next, outer-to-inner.

- The most relevant recent journal entries go next.

The exact policy on what blocks we use, how they are ordered, etc. is something we might iterate on a lot, which is why it is valuable to have a clear separation between the different parts of the mechanism: deciding what directories are relevant, identifying relevant context in each directory, and assembling the system prompt from the relevant context that was identified/collected.

Note that the collection of tools that are presented as part of the completion request also needs to receive the same representation of relevant context directories (so it can identify things like MCP-based tools, or any other tools loaded from relevant context paths).
This isn't part of the construction of the system prompt itself, but it a related piece of policy that should use the same source of truth for what context is relevant.