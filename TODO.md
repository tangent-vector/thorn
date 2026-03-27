
- MCP support, in two senses:
  - Allow MCP servers and their tools to be added to the suite of stuff that can be exposed to skills/roles
  - Allow a module with thorn-based skills to be served via the `mcp` package

- Consume typical definitions of skills, slash commands and personas (e.g., like in `.claude/`)

- Allow `tools=` to support iterables of tools in the list alongside individual tools, so that users can easily write a shorthand for a list/set of tools. Thorn should probably expose basic `file_reading` (read files, list directories, grep, globbing search), `file_manipulation` (`file_reading` plus the ability to write files and create directories), `web_research` (web searches via something like duckduckgo, plus beautifulsoup or similar for extracting the content).

- Provide a centralized mechanism for file permissions management, so that agents roles can include explicit opt-in or opt-out access to specific files, directories, etc. (probably using notation similar to `.gitignore`). All file access through Thorn's built-in file toolset should check for permissions according to those rules.

  Note that having read-write *permissions* to a file path doesn't mean writing is automatically possible, since an agent also needs access to the `write_file` tool.

  The default `Agent` class should probably provide a default of write access to `.` (meaning the "workspace" directory, whatever that should be). (The concept of the "workspace" directory for Thorn should probably default to the CWD at the time `thorn` was launched, but its also possible it should support defaulting to the deepest enclosing directory with a `.thorn/` directory under it)

- Thorn should probably read and respect any `AGENTS.md` file(s) that are set up in a project, using the conventions established by other tools. The content of those files should be piped into `thorn`s agents as additional system-prompt content.
